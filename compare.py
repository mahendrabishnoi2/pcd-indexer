import json
import os
from deepdiff import DeepDiff


def main(dir1, dir2):
    for filename in os.listdir(dir1):
        if not os.path.isfile(os.path.join(dir1, filename)):
            main(os.path.join(dir1, filename), os.path.join(dir2, filename))
        else:
            if not os.path.exists(os.path.join(dir2, filename)):
                print(f"{filename} exists in {dir1} but not in {dir2}")
                continue
            # parse files as json
            file1 = os.path.join(dir1, filename)
            file2 = os.path.join(dir2, filename)

            data1 = {}
            data2 = {}

            with open(file1) as f1:
                data1 = json.load(f1)
            with open(file2) as f2:
                data2 = json.load(f2)

            diff = DeepDiff(data1, data2, significant_digits=3, ignore_numeric_type_changes=True)
            if len(diff) > 0:
                print(f"diff in \n{file1}\n and \n{file2}\n diff: {diff.to_json()}")
                raise Exception('Different files')


if __name__ == '__main__':
    d1 = "/home/mbish2/work/py/pcd-indexer/pcds/pcdFile_219350167"
    d2 = "/home/mbish2/work/py/pcd-indexer/generated-data/final_pcd_files_556434522"
    main(d1, d2)
