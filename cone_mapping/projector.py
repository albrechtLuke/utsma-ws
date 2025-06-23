def project_to_3d(point_cloud, u, v):
    X, Y, Z, _ = point_cloud.get_value(u, v)
    return X, Y, Z