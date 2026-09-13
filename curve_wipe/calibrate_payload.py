"""Interactive manual-pose gravity calibration. No robot motion commands.

collect: manually pose, release tool, press Enter; save train/holdout samples.
fit: estimate sensor rotation, mass, COM and six biases from stationary data.
Sensor origin translation must be mechanically measured; gravity cannot identify it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .capture import read_state
from .execute import rigid_transform, rotation_angle_deg
from .force import compensate_wrench
from .units import WRENCH_UNITS

G=np.array([0.,0.,-9.80665])


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    tmp.replace(path)


def pose(state):
    return rigid_transform(np.array(state['O_T_EE']).reshape(4,4,order='F'),'O_T_EE')


def check_static(states):
    transforms=[pose(s) for s in states]
    if max(np.linalg.norm(t[:3,3]-transforms[0][:3,3]) for t in transforms)>.0005:
        raise ValueError('机械臂移动超过0.5mm，本组丢弃')
    if max(rotation_angle_deg(t[:3,:3],transforms[0][:3,:3]) for t in transforms)>.3:
        raise ValueError('机械臂姿态变化超过0.3度，本组丢弃')
    if np.max(np.abs([s['dq'] for s in states]))>.01:
        raise ValueError('关节尚未静止，本组丢弃')
    if any(not np.allclose(s['F_T_EE'],states[0]['F_T_EE'],atol=1e-6) for s in states):
        raise ValueError('EE定义发生变化')
    return transforms[len(transforms)//2]


def capture_sample(ip,seconds):
    from .serial_sensor import KunweiSensor
    # Open/close per pose so manual positioning is not coupled to a capture.
    with KunweiSensor() as sensor:
        time.sleep(.3)
        states=[read_state(ip)];rows=[];last_frame=0
        begun=time.monotonic();middle=False;bad0=sensor.stats['bad']
        while time.monotonic()-begun<seconds:
            s=sensor.wait_next(last_frame,timeout_s=.1,max_age_s=.05)
            last_frame=s.frame
            rows.append([s.received_monotonic_s,*s.raw_sensor_6])
            if not middle and time.monotonic()-begun>=seconds/2:
                states.append(read_state(ip));middle=True
        states.append(read_state(ip))
        if sensor.stats['bad']!=bad0:raise ValueError('采样期间有无效串口帧，本组丢弃')
    T=check_static(states);a=np.array(rows)
    if len(a)<100:raise ValueError('有效样本不足100帧')
    raw=a[:,1:];std=raw.std(axis=0)
    if np.max(std[:3])>.05 or np.max(std[3:])>.005:
        raise ValueError('力信号波动过大：请松手、等待停稳，检查线缆是否牵拉')
    return dict(time_utc=datetime.now(timezone.utc).isoformat(),T_base_ee=T.tolist(),
                F_T_EE=np.array(states[0]['F_T_EE']).reshape(4,4,order='F').tolist(),
                raw_mean_6=raw.mean(axis=0).tolist(),raw_std_6=std.tolist(),
                raw_samples=rows,robot_states=states,
                acquisition='stationary robot states before/mid/after; host monotonic force receipt, not hardware synchronized')


def collect(args):
    output=Path(args.output)
    if output.exists():
        data=json.loads(output.read_text())
        if data.get('wrench_units') != WRENCH_UNITS:
            raise ValueError('旧会话单位未转换，禁止混合采样；请指定新的 --output 文件')
        if data.get('robot_ip')!=args.ip:raise ValueError('已有会话IP不匹配')
        print('继续已有采样会话。')
    else:data=dict(schema_version=2,wrench_units=WRENCH_UNITS,robot_ip=args.ip,train=[],holdout=[],stationary_free_space_user_confirmed=False)
    print('本程序只读取机器人和传感器，不移动、不清零、不改负载。')
    print('工具必须悬空；夹爪/擦子/线缆不能碰环境，采样时手必须松开。')
    print('手动调整后释放引导按钮，停稳并确保FCI可读，再按Enter采样。q保存退出。')
    confirmation=input('确认安装不变、工具完全悬空，输入 YES 开始：').strip()
    if confirmation!='YES':return
    data['stationary_free_space_user_confirmed']=True
    for role,count in [('train',args.train),('holdout',args.holdout)]:
        if role=='holdout':print('\n现在采独立验证姿态：换成未用于拟合的新倾角，不能重复训练姿态。')
        while len(data[role])<count:
            answer=input(f'[{role} {len(data[role])+1}/{count}] 换到不同倾斜姿态、松手停稳后按Enter（q退出）：').strip().lower()
            if answer=='q':save(output,data);return
            try:
                sample=capture_sample(args.ip,args.seconds)
                all_samples=data['train']+data['holdout']
                if all_samples and not np.allclose(sample['F_T_EE'],all_samples[0]['F_T_EE'],atol=1e-6):
                    raise ValueError('本会话EE定义改变，需新建会话')
                u=np.array(sample['T_base_ee'])[:3,:3].T@G/np.linalg.norm(G)
                for previous in all_samples:
                    v=np.array(previous['T_base_ee'])[:3,:3].T@G/np.linalg.norm(G)
                    if np.degrees(np.arccos(np.clip(u@v,-1,1)))<3:
                        raise ValueError('重力方向与已有姿态过近(<3°)，请绕另一个方向倾斜；只绕竖直轴转动无效')
                data[role].append(sample);save(output,data)
                print('已保存；力均值 N：',np.round(sample['raw_mean_6'][:3],4), '力标准差 N：',np.round(sample['raw_std_6'][:3],4))
            except (ValueError,RuntimeError,OSError,TimeoutError) as exc:print('本组未保存：',exc)
            except Exception as exc:print('读取失败（检查FCI和串口占用），本组未保存：',exc)
    print(f'完成，数据保存在 {output}；下一步运行 fit。')


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def fit_samples(train,origin_mm):
    if len(train)<12:raise ValueError('至少12个训练姿态')
    origin=np.asarray(origin_mm,float)/1000
    if origin.shape!=(3,) or not np.isfinite(origin).all() or np.linalg.norm(origin)>.5:
        raise ValueError('传感器原点需为EE坐标系下的实测毫米值，且距离不超过500mm')
    Ts=[rigid_transform(s['T_base_ee'],'sample pose') for s in train]
    raw=np.array([s['raw_mean_6'] for s in train],float)
    if raw.shape!=(len(train),6) or not np.isfinite(raw).all():raise ValueError('无效六维数据')
    u=np.array([T[:3,:3].T@G for T in Ts])
    centered=u-u.mean(axis=0);sv=np.linalg.svd(centered,compute_uv=False)
    if sv[-1]<.15 or sv[0]/sv[-1]>30:
        raise ValueError('姿态激励不足：需两个方向的正负倾斜及组合倾斜，不能只有一条旋转轴')
    design=np.c_[u,np.ones(len(u))]
    A=np.linalg.lstsq(design,raw[:,:3],rcond=None)[0][:3].T
    sign=1 if np.linalg.det(A)>0 else -1
    U,S,Vt=np.linalg.svd(sign*A);R_sensor_ee=U@Vt
    if np.linalg.det(R_sensor_ee)<0 or min(S)<=0 or max(S)/min(S)>1.5:
        raise ValueError('力模型不符合刚体重力：检查接触、单位、数据或增加姿态跨度')
    initial=np.r_[Rotation.from_matrix(R_sensor_ee).as_rotvec(),np.log(np.mean(S)),raw[:,:3].mean(0)-sign*np.mean(S)*(u@R_sensor_ee.T).mean(0)]
    def residual(x):
        R=Rotation.from_rotvec(x[:3]).as_matrix();mass=np.exp(x[3])
        return (sign*mass*(u@R.T)+x[4:7]-raw[:,:3]).ravel()
    fit=least_squares(residual,initial,max_nfev=3000)
    if not fit.success:raise ValueError('重力拟合未收敛')
    Rse=Rotation.from_rotvec(fit.x[:3]).as_matrix();mass=float(np.exp(fit.x[3]))
    if not .01<mass<10:raise ValueError('拟合质量不合理，检查N单位与安装')
    g_sensor=u@Rse.T
    B=np.vstack([np.c_[-skew(g),np.eye(3)] for g in g_sensor])
    if np.linalg.matrix_rank(B)<6:raise ValueError('质心拟合姿态退化')
    q=np.linalg.lstsq(B,(sign*raw[:,3:]).ravel(),rcond=None)[0]
    com=q[:3]/mass
    if np.linalg.norm(com)>.5:raise ValueError('拟合质心距传感器>500mm，拒绝结果')
    T=np.eye(4);T[:3,:3]=Rse.T;T[:3,3]=origin
    return dict(status='candidate_requires_independent_validation',wrench_units=WRENCH_UNITS,T_ee_sensor=T.tolist(),
                bias_sensor_6=np.r_[fit.x[4:7],sign*q[3:]].tolist(),gravity_base_N=(mass*G).tolist(),
                com_sensor_m=com.tolist(),sensor_sign=sign,mass_kg=mass,
                excitation_singular_values=sv.tolist(),excitation_condition=float(sv[0]/sv[-1]),
                sensor_origin_source='user mechanically measured EE coordinates',
                mounting_rotation_source='joint gravity fit; requires multi-axis excitation')


def evaluate(calibration,samples):
    if not samples:raise ValueError('无验证数据')
    params={k:calibration[k] for k in ['T_ee_sensor','bias_sensor_6','gravity_base_N','com_sensor_m','sensor_sign']}
    forces=[];moments=[]
    for s in samples:
        w=compensate_wrench(s['raw_mean_6'],T_base_ee=s['T_base_ee'],T_ee_tcp=calibration['T_ee_sensor'],**params)
        forces.append(np.linalg.norm(w.force_base_N));moments.append(np.linalg.norm(w.torque_at_tcp_base_Nm))
    return dict(force_rms_N=float(np.sqrt(np.mean(np.square(forces)))),force_max_N=float(max(forces)),
                torque_rms_Nm=float(np.sqrt(np.mean(np.square(moments)))),torque_max_Nm=float(max(moments)),
                per_pose_force_norm_N=forces,per_pose_torque_norm_Nm=moments)


def fit_file(args):
    source=Path(args.samples);data=json.loads(source.read_text())
    if data.get('wrench_units') != WRENCH_UNITS:
        raise ValueError('采样单位未确认为 N/Nm，必须先转换旧数据')
    calibration=fit_samples(data['train'],args.sensor_origin_ee_mm)
    all_samples=data['train']+data.get('holdout',[])
    if any(not np.allclose(s['F_T_EE'],all_samples[0]['F_T_EE'],atol=1e-6) for s in all_samples):
        raise ValueError('采样期间EE定义不一致')
    calibration['training_report']=evaluate(calibration,data['train'])
    holdout=data.get('holdout',[])
    independent=len(holdout)>=5
    if holdout:
        for i,s in enumerate(holdout):
            u=np.array(s['T_base_ee'])[:3,:3].T@G/np.linalg.norm(G)
            for t in data['train']+holdout[:i]:
                v=np.array(t['T_base_ee'])[:3,:3].T@G/np.linalg.norm(G)
                independent &= np.degrees(np.arccos(np.clip(u@v,-1,1)))>=3
        calibration['holdout_report']=evaluate(calibration,holdout)
    def passed(r):return r['force_rms_N']<=.1 and r['force_max_N']<=.2 and r['torque_max_Nm']<=.03
    ok=(independent and data.get('stationary_free_space_user_confirmed') is True and
        passed(calibration['training_report']) and passed(calibration.get('holdout_report',dict(force_rms_N=999,force_max_N=999,torque_max_Nm=999))))
    if ok:calibration['status']='validated_current_mount'
    calibration.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       source_samples=str(source.resolve()),F_T_EE=all_samples[0]['F_T_EE'],
                       calibration_time_utc=datetime.now(timezone.utc).isoformat(),
                       limitations=['Static gravity model only; no cable/inertial compensation.',
                                    'Valid only while tool, sensor mounting, EE definition and units remain unchanged.',
                                    'Origin translation is user supplied and not identifiable from gravity.'])
    save(args.output,calibration)
    print(json.dumps({k:v for k,v in calibration.items() if k in ['status','mass_kg','com_sensor_m','sensor_sign','training_report','holdout_report']},indent=2))
    print('标定通过，可用于当前安装。' if ok else '仅保存候选结果，未通过验证，不可用于变姿态接触执行。')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect',help='交互采样，只读机器人，不运动')
    c.add_argument('--ip',default='172.16.0.2');c.add_argument('--output',default='data/payload_calibration/session.json')
    c.add_argument('--train',type=int,default=16);c.add_argument('--holdout',type=int,default=6);c.add_argument('--seconds',type=float,default=2.)
    f=sub.add_parser('fit',help='离线拟合及独立验证，不连接硬件')
    f.add_argument('--samples',default='data/payload_calibration/session.json')
    f.add_argument('--sensor-origin-ee-mm',type=float,nargs=3,required=True,metavar=('X','Y','Z'))
    f.add_argument('--output',default='config/gravity_calibration.json')
    args=parser.parse_args()
    if args.command=='collect':
        if args.train<12 or args.holdout<5 or not 1<=args.seconds<=10:parser.error('至少12训练/5验证姿态；采样时长1..10秒')
        collect(args)
    else:fit_file(args)


if __name__=='__main__':main()
