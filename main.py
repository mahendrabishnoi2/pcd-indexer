from loader import parse_point_cloud

if __name__ == '__main__':
    file_path_large = '/Users/mahendrabishnoi/workspace/pcd-indexer/scan-14_Sample-1pixels_ascii.pcd'
    file_path_medium = '/Users/mahendrabishnoi/Downloads/b2fa861b-e214-48eb-b4b5-54a8b0833866.pcd'
    file_path_small = '/Users/mahendrabishnoi/Downloads/PC_315970951220244000.pcd'
    parse_point_cloud(file_path_small)
