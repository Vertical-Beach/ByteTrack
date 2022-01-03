import json
import argparse
import glob
import os


VIDEO_NUM = 74
FRAME_WIDTH = 1936
FRAME_HEIGHT = 1216
TEST_VIDEO_FRAME_NUM = 150 #5fps * 30sec

class Bbox:
    def __init__(self, line):
        elems = line.split(',')
        elems = [int(float(x.strip())) for x in elems]
        self.frame_id = elems[0]
        self.object_id = elems[1]
        self.category = elems[2]
        # x1, y1, w, h
        self.x1 = max(0, elems[3])
        self.y1 = max(0, elems[4])
        self.x2 = min(FRAME_WIDTH, elems[3] + elems[5])
        self.y2 = min(FRAME_HEIGHT, elems[4] + elems[6])

    def to_bboxdic(self):
        return {
            "id": self.object_id,
            "box2d": [self.x1, self.y1, self.x2, self.y2]
        }
    @staticmethod
    def convert_line_to_bboxes(lines):
        return [Bbox(line) for line in lines]

def count_tracked_frames(bboxes):
    count_dic = {}
    for bbox in bboxes:
        key = (bbox.object_id, bbox.category)
        if key not in count_dic:
            count_dic[key] = 0
        count_dic[key] += 1
    return count_dic

def remove_few_frame_bboxes(count_dic, bboxes):
    new_bboxes = []
    for bbox in bboxes:
        key = (bbox.object_id, bbox.category)
        if count_dic[key] < 3:
            print(f"Removed object_id={bbox.object_id}, category={bbox.category}")
        else:
            new_bboxes.append(bbox)
    return new_bboxes

def bboxes_to_dic(bboxes):
    # readme.txt on SIGNATE says:
    # If you do not want to make any prediction in some frames, just write "{}" in the corresponding frames.

    result_list = [{} for i in range(TEST_VIDEO_FRAME_NUM)]
    for bbox in bboxes:
        assert(bbox.frame_id < TEST_VIDEO_FRAME_NUM)
        category_name = ["Car", "Pedestrian"][bbox.category]
        if category_name not in result_list[bbox.frame_id]:
            result_list[bbox.frame_id][category_name] = []
        result_list[bbox.frame_id][category_name].append(bbox.to_bboxdic())
    return result_list

def process(txt_file_path):
    bboxes = Bbox.convert_line_to_bboxes(open(txt_file_path).readlines())
    # 1. count number of tracked frames for each tracked id
    count_dic = count_tracked_frames(bboxes)
    # 2. remove bboxes whose tracked frame num is less than 3
    bboxes = remove_few_frame_bboxes(count_dic, bboxes)
    # 3. convert to list of dictionary
    result_list = bboxes_to_dic(bboxes)
    return result_list

def main(args):
    txt_file_paths = glob.glob(os.path.join(args.result_dir, "*/test_*.txt"))
    txt_file_paths = sorted(txt_file_paths)
    assert(len(txt_file_paths) == VIDEO_NUM)

    result_dic = {}
    for txt_file_path in txt_file_paths:
        video_name = os.path.basename(txt_file_path).replace(".txt", ".mp4")
        result_dic[video_name] = process(txt_file_path)

    open(args.output_path, "w").write(json.dumps(result_dic))
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--output_path", type=str, default="./submit.json")
    args = parser.parse_args()
    main(args)