"""
trajectory_prediction.py

简易轨迹预测：常速度（CV）与常加速度（CA）模型。
接口：predict_future(traj, horizon_sec, dt, model='cv'|'ca')

输入 traj 为由 ObjectStateManager 返回的样本列表，每项包含 {x,y,t}
返回未来时间点的预测序列 list of {t,x,y}
"""
from typing import List, Dict


def predict_future(traj: List[Dict], horizon_sec: float = 1.0, dt: float = 0.1, model: str = 'cv') -> List[Dict]:
    if len(traj) < 1:
        return []
    # 使用最后两个点估计速度，若只有一个点则假设静止
    if len(traj) >= 2:
        p0 = traj[-2]
        p1 = traj[-1]
        dt_hist = p1['t'] - p0['t'] if (p1['t'] - p0['t']) != 0 else 1e-6
        vx = (p1['x'] - p0['x']) / dt_hist
        vy = (p1['y'] - p0['y']) / dt_hist
    else:
        vx = 0.0
        vy = 0.0
    if model == 'ca':
        # 估计加速度（若有 >=3 点），否则为 0
        if len(traj) >= 3:
            p_2 = traj[-3]
            dt1 = p0['t'] - p_2['t'] if (p0['t'] - p_2['t']) != 0 else 1e-6
            vx0 = (p0['x'] - p_2['x']) / dt1
            vy0 = (p0['y'] - p_2['y']) / dt1
            ax = (vx - vx0) / dt_hist
            ay = (vy - vy0) / dt_hist
        else:
            ax, ay = 0.0, 0.0
    else:
        ax, ay = 0.0, 0.0

    preds = []
    last_t = traj[-1]['t']
    last_x = traj[-1]['x']
    last_y = traj[-1]['y']
    steps = int(horizon_sec / dt)
    for i in range(1, steps + 1):
        tt = last_t + i * dt
        # s = s0 + v * dt + 0.5 * a * dt^2
        dt_tot = tt - last_t
        x = last_x + vx * dt_tot + 0.5 * ax * (dt_tot ** 2)
        y = last_y + vy * dt_tot + 0.5 * ay * (dt_tot ** 2)
        preds.append({'t': tt, 'x': float(x), 'y': float(y)})
    return preds


if __name__ == '__main__':
    # 简单自测
    traj = [{'x': 0, 'y': 0, 't': 0}, {'x': 1, 'y': 0, 't': 1}]
    print(predict_future(traj, horizon_sec=2.0, dt=0.5, model='cv'))
