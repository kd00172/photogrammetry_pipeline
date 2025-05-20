import cv2
import numpy as np
import py360convert as p360

# 1. 입력 equirectangular 이미지 읽기
inp_path  = './objects_only.png'
equi      = cv2.imread(inp_path, cv2.IMREAD_UNCHANGED)

# 2. 투영 설정: 시야각(fov), 출력 해상도, 회전 각도(yaw, pitch, roll)
fov       = 160            # degrees
out_w     = 1024          # 출력 너비
out_h     = 768           # 출력 높이
yaw       =   260           # 좌우 회전, 양수 = 오른쪽 (degrees)
pitch     =   0          # 상하 회전, 양수 = 위쪽 (degrees)
roll      =   0           # 회전 (degrees)

# 3. e2p 호출: equirectangular → perspective (알파채널 보존)
if equi.ndim == 3 and equi.shape[2] == 4:
    bgr   = equi[..., :3]
    alpha = equi[...,  3]
    persp_bgr = p360.e2p(
        bgr,
        fov_deg=fov,
        u_deg=yaw,
        v_deg=pitch,
        out_hw=(out_h, out_w),
        in_rot_deg=roll,
        mode='bilinear'
    )
    persp_alpha = p360.e2p(
        alpha[..., None],
        fov_deg=fov,
        u_deg=yaw,
        v_deg=pitch,
        out_hw=(out_h, out_w),
        in_rot_deg=roll,
        mode='nearest'
    )
    persp = np.dstack((persp_bgr, persp_alpha[...,0]))
else:
    persp = p360.e2p(
        equi,
        fov_deg=fov,
        u_deg=yaw,
        v_deg=pitch,
        out_hw=(out_h, out_w),
        in_rot_deg=roll,
        mode='bilinear'
    )

# 4. 결과 저장
out_path = 'perspective_view.png'
cv2.imwrite(out_path, persp)
print(f"저장 완료: {out_path}")