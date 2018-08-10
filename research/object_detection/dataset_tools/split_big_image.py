# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import json
import errno
import PIL.Image
import tensorflow as tf
import copy

flags = tf.app.flags
flags.DEFINE_string('src_dir',
                    "/home/keyong/Downloads/18001000W_1102",
                    """the source folder which contains image and annotatoin should have structure 
.
└── Photos
      └── Annotations""")
flags.DEFINE_string('to_dir',
                    "/home/keyong/Downloads/splited_data",
                    'the destination folder to store splitted pictures and annotations')
flags.DEFINE_boolean('split',True, 'split a dataset(for small object) or simply scale big image at ratio 0.25'
                    'merged set.')

FLAGS = flags.FLAGS


# {"version":"1.0.0","company":"18001000W","dataset":"Photos",
#  "filename":"1002_3428_IMG_5944.JPG",
#  "image_width":3024,"image_height":4032,
#  "bndboxes":[
#    {"x":242.98105263157896,"y":1638.2652631578949,"w":234.43157894736845,"h":698.2616673007506,"id":"235402",
#           "strokeStyle":"#3399FF","fillStyle":"#00FF00"},
#    {"x":315.1326315789474,"y":2544.889710336091,"w":213.21052631578942,"h":550.1397633481197,"id":"222285",
#             "strokeStyle":"#3399FF","fillStyle":"#00FF00"},
#  ]}
#
def bbox_in_area(bbox,x,y,w,h):
    bx=bbox.get("x")
    by=bbox.get("y")
    bw=bbox.get("w")
    bh=bbox.get("h")
    threshold=0.75
    rx,ry,rx2,ry2= (max(bx,x), max(by,y), min(bx+bw,x+w) ,min(by+bh,y+h))
    if rx2<=rx or ry2<=ry:
        return None

    if (rx2-rx) * (ry2 -ry) >= bw*bh* threshold:
        new_box=copy.deepcopy(bbox)
        new_box["x"]=rx
        new_box["y"]=ry
        new_box["w"]=rx2-rx
        new_box["h"]=ry2-ry
        return new_box

    return None


def crop_image(x, y, w, h, row, col, scaled_img, to_dir, short_name, json_data):
    #crop = "-crop 500x500+" + str(x) + "+" + str(y)
    crop = "-crop "+str(w)+"x"+str(h)+"+" + str(x) + "+" + str(y)
    #print (crop)
    photo_dir = os.path.join(to_dir, "photos")

    cropped_image_name = short_name[0:-4] + "_" + str(row) + "_" + str(col) + ".JPG"
    photo_name = os.path.join(photo_dir, cropped_image_name)
    command = "convert '" + scaled_img + "' " + crop + " '" + photo_name+"'"
    if not os.path.exists(photo_name):
        os.system(command)
    avai_bbs=[]
    for box in json_data.get("bndboxes"):
        new_box = bbox_in_area(box,x,y,w,h)
        if new_box is not None:
            avai_bbs.append(new_box)
        # if box.get("x") >= x and box.get("x") + box.get("w") < x + w:
        #     by=box["y"]
        #     bh=box["h"]
        #     if by >= y and by + bh <= y + h:
        #         avai_bbs.append(copy.deepcopy(box))
        #     elif by <y and by + bh > y:
        #         #box is upper of crop area
        #         overlap= by+bh -y
        #         if overlap >= bh*threshold:
        #             new_box=copy.deepcopy(box)
        #             new_box["y"]=y
        #             new_box["h"]=bh+by-y
        #             avai_bbs.append(new_box)
        #         pass
        #     elif by < y+h and by + bh > y + h:
        #         overlap= y + h - by
        #         if overlap >= bh*threshold:
        #             new_box=copy.deepcopy(box)
        #             new_box["h"]=y+h-by
        #             avai_bbs.append(new_box)
        # elif box.get("y") >= y and box.get("y") + box.get("h") < y + h:
        #     bx=box["x"]
        #     bw=box["w"]
        #
        #     if bx < x and bx + bw > x:
        #         overlap= bx+bw -x
        #         if overlap >= bw*threshold:
        #             new_box=copy.deepcopy(box)
        #             new_box["x"]=x
        #             new_box["w"]=bx+bw-x
        #             avai_bbs.append(new_box)
        #     elif bx < x+w and bx + bw > x + w:
        #         overlap= x + w - bx
        #         if overlap >= bw*threshold:
        #             new_box=copy.deepcopy(box)
        #             new_box["w"]= x + w -bx
        #             avai_bbs.append(new_box)
        #pass

    for box in avai_bbs:
        box["x"] = box["x"] - x
        box["y"] = box["y"] - y


    with open(os.path.join(to_dir, "photos", "Annotations", cropped_image_name + ".json"),'w') as fp:
        header = '{"version":"1.0.0","company":"idontknow","dataset":"photos","filename":"'
        header +=  cropped_image_name + '",' + """
  "image_width":%d,"image_height":%d,
  "bndboxes":
 """%(w,h)
        footer = "}"
        fp.write(header)
        bbs = json.dumps(avai_bbs)
        fp.write(bbs)
        fp.write(footer)


def scale_one_image(json_data, img_path, to_dir, short_name):
    bndboxes = json_data.get("bndboxes")
    scale=0.25
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        with PIL.Image.open(encoded_jpg_io) as image:
            width, height = image.size

    new_width = int(width * scale)
    new_height = int(height * scale)

    if new_width < 500 :
        scale = 500.0/width
        new_width = int(width * scale)
        new_height = int(height * scale)

    if new_height< 500 :
        scale = 500.0/height
        new_width = int(width * scale)
        new_height = int(height * scale)

    for box in bndboxes:
        box["x"]=box["x"]*scale
        box["y"]=box["y"]*scale
        box["w"]=box["w"]*scale
        box["h"]=box["h"]*scale

    orig_full_path = img_path
    # short_name is the file name like "abc.JPG"
    bgr=False

    temp_img_rgb = os.path.join(to_dir, "photos",  short_name+"rgb.jpg")
    temp_img = os.path.join(to_dir, "photos",  short_name)

    #slice_img = os.path.join(to_dir, "Photos", short_name[0:-4])
    if not os.path.exists(temp_img):
        if bgr:
            os.system("convert '" + orig_full_path + "' -resize " + str(new_width) + "x" + str(new_height) + " '" + temp_img_rgb+"'")
            #to_bgr:convert 1.jpg  -separate +channel -swap 0,2  -combine 1_bgr.jpg
            os.system("convert '" + temp_img_rgb + "' -separate +channel -swap 0,2 -combine '" + temp_img+"'")
        else:
            os.system("convert '" + orig_full_path + "' -resize " + str(new_width) + "x" + str(new_height) + " '" + temp_img+"'")

    with open(os.path.join(to_dir, "photos", "Annotations", short_name + ".json"),'w') as fp:
        header = '{"version":"1.0.0","company":"idontknow","dataset":"photos","filename":"'
        header +=  short_name + '",' + """
  "image_width":500,"image_height":500,
  "bndboxes":
 """
        footer = "}"
        fp.write(header)
        bbs = json.dumps(bndboxes)
        fp.write(bbs)
        fp.write(footer)

def split_one_image(json_data, img_path, to_dir, short_name):
    bndboxes = json_data.get("bndboxes")
    min_w = min(bndboxes, key=lambda bbox: bbox.get("w")).get("w")
    min_h = min(bndboxes, key=lambda bbox: bbox.get("h")).get("h")

    if min_w < 45:
        error_str = "Wrong bounding box in " + img_path + ", the width of bbox is " + str(
            min_w) + ", should be bigger than 45"
        print(error_str)
        raise ValueError(error_str)
    try:
        scale = 60.0 / min_w
    except:
        print("type(min_w)="+str(type(min_w) )+", min_w="+min_w)
    if scale>1.0:
        scale=1.0
    print("scale level is %f" %(scale))
    tile_size=600
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    with PIL.Image.open(encoded_jpg_io) as image:
        width, height = image.size

    new_width = int(width * scale)
    new_height = int(height * scale)

    if new_width < tile_size :
        scale = tile_size*1.0/width
        new_width = int(width * scale)
        new_height = int(height * scale)

    if new_height< tile_size :
        scale = tile_size*1.0/height
        new_width = int(width * scale)
        new_height = int(height * scale)

    for box in bndboxes:
        box["x"]=box["x"]*scale
        box["y"]=box["y"]*scale
        box["w"]=box["w"]*scale
        box["h"]=box["h"]*scale


    orig_full_path = img_path
    # short_name is the file name like "abc.JPG"
    bgr=False

    temp_img_rgb = os.path.join(to_dir, "photos", "tmp", short_name+"rgb.jpg")
    temp_img = os.path.join(to_dir, "photos", "tmp", short_name)

    slice_img = os.path.join(to_dir, "photos", short_name[0:-4])
    if not os.path.exists(temp_img):
        if bgr:
            os.system("convert '" + orig_full_path + "' -resize " + str(new_width) + "x" + str(new_height) + " '" + temp_img_rgb+"'")
            #to_bgr:convert 1.jpg  -separate +channel -swap 0,2  -combine 1_bgr.jpg
            os.system("convert '" + temp_img_rgb + "' -separate +channel -swap 0,2 -combine '" + temp_img+"'")
        else:
            os.system("convert '" + orig_full_path + "' -resize " + str(new_width) + "x" + str(new_height) + " '" + temp_img+"'")

    y = 0
    row = 0

    while y + tile_size < new_height:
        col = 0
        x = 0
        #new_tile_height = get_recommended_tile_heigh(x,y, bndboxes)
        while x + tile_size < new_width:
            crop_image(x, y, tile_size, tile_size, row, col, temp_img, to_dir, short_name, json_data)
            x += tile_size*2/3
            col += 1

        if x != new_width:
            crop_image(new_width - tile_size, y, tile_size, tile_size, row, col, temp_img, to_dir, short_name, json_data)
        row += 1
        y += tile_size*2/3

    if y != new_height:
        col = 0
        y = new_height - tile_size
        x = 0
        while x + tile_size < new_width:
            crop_image(x, y, tile_size, tile_size, row, col, temp_img, to_dir, short_name, json_data)
            x += tile_size*2/3
            col += 1

        if x != new_width:
            crop_image(new_width - tile_size, y, tile_size, tile_size, row, col, temp_img, to_dir, short_name, json_data)

    pass


def mkdir_p(path):
    try:
        os.makedirs(path, 0o770)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def main(_):
    #print("split flag = "+str(FLAGS.split))
    #return
    mkdir_p(os.path.join(FLAGS.to_dir, "photos", "Annotations"))
    mkdir_p(os.path.join(FLAGS.to_dir, "photos", "tmp"))
    src_dir = FLAGS.src_dir
    #print("FLGS.split="+FLAGS.split)
    annotations_dir = os.path.join(src_dir, "photos", "Annotations")
    photos_dir = os.path.join(src_dir, "photos")
    examples_list = [f for f in os.listdir(annotations_dir)
                     if f.find(".scaled.") == -1 and not f.startswith(".") and f.endswith(".json") and os.path.isfile(
            os.path.join(annotations_dir, f))
                     ]
    print("split/scale flag is :%s\n" %( FLAGS.split ))
    for idx, example in enumerate(examples_list):
        if (idx+1) % 100 == 0:
            logging.info('On image %d of %d', idx+1, len(examples_list))
            print('On image %d of %d' %( idx+1, len(examples_list)))

        # path = os.path.join(annotations_dir, example + '.json')
        short_name = example[0:-5]
        path = os.path.join(annotations_dir, example)
        img_path = os.path.join(photos_dir, short_name)
        with tf.gfile.GFile(path, 'r') as fid:
            json_str = fid.read()
        #print("loading json file:%s\n" %(img_path))
        try:
            json_data = json.loads(json_str)
        except:
            print("loading json file:%s\n" %(img_path))
        #if idx < 2700:
        #    continue
        bndboxes = json_data.get("bndboxes")
        bndboxes=[box for box in bndboxes if box.get("w")>=45 ]
        #bndboxes=[box for box in bndboxes if box.get("id") !="160611" and box.get("id") !="170326" ]
        #ooo=[box for box in bndboxes if box.get("id") in ["Amber","Silver","Blue","Green"] ]
        for box in bndboxes:
            #if type(box["w"]) is str:
            box["w"]=float(box["w"])
            #if type(box["h"]) is str:
            box["h"]=float(box["h"])
            #if type(box["x"]) is str:
            box["x"]=float(box["x"])
            #if type(box["y"]) is str:
            box["y"]=float(box["y"])

        json_data["bndboxes"]=bndboxes
        #if len(bndboxes) <= 0:
        #    continue
        #bndboxes = json_data.get("bndboxes")
        #for box in bndboxes:
        #    assert box["id"] not in ["Silver","Blue","Green","unknown"]
        #print("FLAGS.split:%s\n" %(str(FLAGS.split)))
        if FLAGS.split and len(bndboxes)>0:
            #print("split")
            split_one_image(json_data, img_path, FLAGS.to_dir, short_name)
        else:
            #print("scale")
            scale_one_image(json_data, img_path, FLAGS.to_dir, short_name)
        #1002_3428_IMG_5944_2_1if short_name == 'IMG_1303.JPG':
        # if short_name in ['1002_3428_IMG_5944.JPG',"1002_3428_IMG_5945.JPG"]:
        #     print(short_name)
        #     split_one_image(json_data, img_path, FLAGS.to_dir, short_name)
        #     #break
        #break
    pass
    #


if __name__ == '__main__':
    tf.app.run()
