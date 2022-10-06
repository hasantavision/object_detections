import glob
import os
import numpy as np
from generate_xml import GenerateXml
import time
import cv2
import onnxruntime as ort


IMAGE_PATH = 'dummy.png'
ONNX_PATH = 'TF225_centernet_inference_graph_5.onnx'
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])


def do_detect(image_bgr):
    # class_name = ["BG", "keropos", "kurokawa_forging", "dakon", "scratch", "hole", "d78", "scratch_ok", "water_droplet", "keropos_casting", "step", "parting_line"]
    class_name = ["BG", "Keropos", "Kurokawa", "Dakon", "Scratch", "hole", "d78", "scratch_ok", "water_droplet", "keropos_casting", "step", "parting_line"]
    ori = image_bgr.copy()
    image = image_bgr.copy()
    #image = clahein(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_onnx = np.expand_dims(image_rgb, 0)
    detections = sess.run(["detection_boxes", "detection_scores", "detection_classes"], {'input_tensor': image_onnx})
    boxes = detections[0][0]
    scores = detections[1][0]
    classes = detections[2][0].astype(int)
    cls_det = []
    bbox = []
    box_arr = []
    for i in range(len(boxes)):
        kelas = classes[i]
        if (scores[i] >= 0.35 and classes[i] == 1 or \
                scores[i] >= 0.3 and classes[i] == 2 or \
                scores[i] >= 0.4 and classes[i] == 3 or \
                scores[i] >= 0.3 and classes[i] == 4 or \
                scores[i] >= 0.3 and classes[i] == 9 or \
                scores[i] >= 0.3 and classes[i] == 10 or \
                scores[i] >= 0.3 and classes[i] == 11 or \
                scores[i] >= 0.3 and classes[i] == 12) and boxes[i][0] < 0.9:
            box = boxes[i] * np.array([image_bgr.shape[0], image_bgr.shape[1], image_bgr.shape[0], image_bgr.shape[1]])
            box_item = {'ymin': int(box[0]), 'xmin': int(box[1]), 'ymax': int(box[2]), 'xmax': int(box[3])}
            dsize = {'x': (box[3] - box[1]) * 67 / 1280, 'y': (box[2] - box[0]) * 67 / 1024}
            box_arr.append(box_item)
            x_text = int(box[1]) - 100
            y_text = int(box[0]) - 10
            if int(box[0] < 30):
                y_text = int(box[2]) + 15
            if int(box[3]) > 1050:
                x_text = int(box[1]) - 300
            elif int(box[1]) < 100:
                x_text = int(box[1])
            image = cv2.rectangle(image, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 0, 255), 2)
            image = cv2.putText(image, '%s %.2f size: (%.2f, %.2f)' % (class_name[int(classes[i])], scores[i], dsize['x'], dsize['y']),
                                (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
            cls_det.append(class_name[kelas])
            bbox.append(box)
    return ori, image, cls_det, bbox, box_arr


def clahein(image):
    gridsize = 8
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def warmup():
    dummy = cv2.imread(IMAGE_PATH)
    do_detect(dummy)

def main():
    dummy = cv2.imread(IMAGE_PATH)
    __, image, dot, __, __ = do_detect(dummy)
    print(dot)
    cv2.imshow('result', image)
    print("-------------------------------------")
    print("PRESS 'ESC' TO PERFORM BENCHMARK TEST WHEN IMAGE APPEARS AND IS IN FOCUS")
    print("-------------------------------------")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    num_samples = 50
    t0 = time.time()
    for i in range(int(num_samples)):
        t2 = time.time()
        _, sun, dots, _, _ = do_detect(dummy)
        print('%f [sec]' % (time.time() - t2), dot)
    t1 = time.time()
    print('Average runtime: %f seconds' % (float(t1 - t0) / num_samples))


def main2():
    img_fold = "/media/mapin/DataAI2/11 MASSPRO Image/part 10767"
    secs = sorted(os.listdir(img_fold))
    for sec in secs:
        i_path = os.path.join(img_fold, sec)
        imgs = sorted(os.listdir(i_path))
        for img in imgs:
            if img.endswith('.png'):
                print(img)
                image = cv2.imread(os.path.join(i_path, img))
                ct = time.time()
                __, imgres, culass, boxess, ssbox = do_detect(image)
                if len(boxess) > 0:
                    cv2.imwrite(os.path.join(i_path, f'{img}99.jpg'), imgres)
                    xml = GenerateXml(ssbox, 1280, 1024, 3, culass, img, i_path)
                    print(i_path)
                    print(img)
                    xml.gerenate_basic_structure()
                imgres = cv2.resize(imgres, (0,0), fx=0.5, fy=0.5)
                cv2.imshow('result', imgres)
                k = cv2.waitKey(1)
                if k == 27:
                    cv2.destroyAllWindows()                        
                    break
                print(f'CT: {time.time() - ct}')


def main4():
    img_fold = "/media/mapin/DataAI1/Part_recheck/06-10-2021/['T1', 11512]"
    i_path = os.path.join(img_fold, sec)
    imgs = sorted(os.listdir(i_path))
    for img in imgs:
        if img.endswith('.png'):
            print(img)
            image = cv2.imread(os.path.join(i_path, img))
            ct = time.time()
            __, imgres, culass, boxess, ssbox = do_detect(image)
            if len(boxess) > 0:
                cv2.imwrite(os.path.join(i_path, f'{img}.jpg'), imgres)
                xml = GenerateXml(ssbox, 1280, 1024, culass, img, i_path)
                print(i_path)
                print(img)
                xml.gerenate_basic_structure()
            imgres = cv2.resize(imgres, (0,0), fx=0.5, fy=0.5)
            cv2.imshow('result', imgres)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()                        
                break                
            print(f'CT: {time.time() - ct}')


def main3():
    img_fold = r"G:\IMG-MASSPRO\2022\06\14-06-2022\D98IN\NG"
    output_path = r"F:\FINAL_DATA\Output"
    part_ids = sorted(os.listdir(img_fold))
    for part_id in part_ids:
        secs_path = os.path.join(img_fold, part_id)
        secs = sorted(os.listdir(secs_path))
        for sec in secs:
            imgs_path = os.path.join(secs_path, sec)
            imgs = sorted(os.listdir(imgs_path))
            for img in imgs:
                if img.endswith('.png'):
                    image = cv2.imread(os.path.join(imgs_path, img))
                    ct = time.time()
                    img_ori, img_res, culass, boxess, ssbox = do_detect(image)
                    if len(boxess) > 0:
                        cv2.imwrite(os.path.join(output_path, f"{img[:-4]}.png"), img_ori)
                        xml = GenerateXml(ssbox, 1280, 1024, 3, culass, img, output_path, 'png')
                        xml.gerenate_basic_structure()
                    print(f'CT: {time.time() - ct}')
                    img_res = cv2.resize(img_res, (0,0), fx=0.5, fy=0.5)
                    cv2.imshow('result', img_res)
                    # time.sleep(1)
                    k = cv2.waitKey(1)
                    if k == 27:
                        cv2.destroyAllWindows()
                        break


def mains():
    img_fold = "/media/mapin/DataAI2/11 MASSPRO Image/28-10-2021/part 11037/sect 1"
    imgs = sorted(os.listdir(img_fold))
    for img in imgs:
        if img.endswith('.png'):
            image = cv2.imread(os.path.join(img_fold, img))
            ct = time.time()
            __, img_res, culass, boxess, ssbox = do_detect(image)
            if len(boxess) > 0:
                cv2.imwrite(os.path.join(img_fold, f"{img}999.jpg"), img_res)
                xml = GenerateXml(ssbox, 1280, 1024, culass, img, img_fold)
                xml.gerenate_basic_structure()
            print(f'CT: {time.time() - ct}')
            img_res = cv2.resize(img_res, (0,0), fx=0.5, fy=0.5)
            cv2.imshow('result', img_res)
            # time.sleep(1)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break


def main_glob():
    imgs = glob.glob(r"G:\2021\12\*\*\*\*\*\*.png")
    print(len(imgs))


def main_pan():
    PATH = r"G:\2021\12"

    list_day = sorted(os.listdir(PATH))
    for day in list_day:
        day_dir = os.path.join(PATH, day)
        list_type = sorted(os.listdir(day_dir))
        for p_type in list_type:
            type_dir = os.path.join(day_dir, p_type + "/NG")
            list_part = sorted(os.listdir(type_dir))
            for part in list_part:
                part_dir = os.path.join(type_dir, part)
                list_sec = sorted(os.listdir(part_dir))
                for sec in list_sec:
                    sec_dir = os.path.join(part_dir, sec)
                    list_img = sorted(os.listdir(sec_dir))
                    for img in list_img:
                        if img.endswith(".png"):
                            # img = os.path.join(sec_dir, img)
                            image = cv2.imread(os.path.join(sec_dir, img))
                            ct = time.time()
                            __, img_res, culass, boxess, ssbox = do_detect(image)
                            if len(boxess) > 0:
                                cv2.imwrite(os.path.join(sec_dir, f"{img}.jpg"), img_res)
                                xml = GenerateXml(ssbox, 1280, 1024, culass, img, sec_dir)
                                xml.gerenate_basic_structure()
                            print(f'CT: {time.time() - ct}')
                            img_res = cv2.resize(img_res, (0, 0), fx=0.5, fy=0.5)
                            cv2.imshow('result', img_res)
                            # time.sleep(1)
                            k = cv2.waitKey(1)
                            if k == 27:
                                cv2.destroyAllWindows()
                                break


if __name__ == '__main__':
    main3()
