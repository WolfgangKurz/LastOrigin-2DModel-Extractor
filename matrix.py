import numpy as np

class Matrix3D:
    def translate(x: float, y: float, z: float):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            x, y, z, 1,
        ]).reshape(4, 4)
    
    def scale(x: float, y: float, z: float):
        return np.array([
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1,
        ]).reshape(4, 4)

    def rotateX(rad: float):
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([
            1, 0, 0, 0,
            0, c, -s, 0,
            0, s, c, 0,
            0, 0, 0, 1,
        ]).reshape(4, 4)

    def rotateY(rad: float):
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([
            c, 0, s, 0,
            0, 1, 0, 0,
            -s, 0, c, 0,
            0, 0, 0, 1,
        ]).reshape(4, 4)

    def rotateZ(rad: float):
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([
            c, -s, 0, 0,
            s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]).reshape(4, 4)

    def rotate(radX: float, radY: float, radZ: float):
        return Matrix3D.compose([
            Matrix3D.rotateX(radX),
            Matrix3D.rotateY(radY),
            Matrix3D.rotateZ(radZ)
        ])

    def compose(m: list[np.ndarray]):
        r = np.identity(4)
        for _m in m:
            r = np.matmul(_m, r)
        return r
