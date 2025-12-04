import numpy as np

def cal_euler_distance(angles1, angles2):
    phi = angles1[0]*np.pi/180
    theta = angles1[1]*np.pi/180
    psi = angles1[2]*np.pi/180
    phi_o = angles2[0]*np.pi/180
    theta_o = angles2[1]*np.pi/180
    psi_o = angles2[2]*np.pi/180
    
    r0 = np.cos(theta/2) * np.cos(theta_o/2) * np.cos((psi+phi)/2) * np.cos((psi_o+phi_o)/2) + np.sin(theta/2)*np.sin(theta_o/2) * np.cos((psi-phi)/2) * np.cos((psi_o-phi_o)/2) + np.sin(theta/2)*np.sin(theta_o/2) * np.sin((psi-phi)/2) * np.sin((psi_o-phi_o)/2) + np.cos(theta/2)*np.cos(theta_o/2) * np.sin((psi+phi)/2) * np.sin((psi_o+phi_o)/2)
    r0 = round(r0, 10) # 浮点数误差
    euler_dist = np.abs((np.arccos(r0)*2*180/np.pi + 180)%360 - 180)
    return euler_dist