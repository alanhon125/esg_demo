import re
import json
import cv2
import glob
import os

def define_rect(image, fname, page_id):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """

    clone = image.copy()
    h, w, _ = clone.shape
    rects = []
    rect_pts = [] # Starting and ending points
    win_name = "image" # Window name
    coord = []
    draw_count = 0

    def select_points(event, x, y, flags, param):
        nonlocal coord
        nonlocal rect_pts
        nonlocal draw_count

        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]
            # normalize coordinates
            x0 = min(1000, max(0, int(x / w * 1000))) #round(x / w * 1000,2)
            y0 = min(1000, max(0, int(y / h * 1000))) #round(y / h * 1000,2)
            coord.append((x0, y0))
            print("Starting point is ",(x,y),", normalized ", (x0,y0))

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))
            # normalize coordinates
            x1 = min(1000, max(0, int(x / w * 1000))) #round(x / w * 1000,2)
            y1 = min(1000, max(0, int(y / h * 1000))) #round(y / h * 1000,2)
            coord.append((x1, y1))
            print("Ending point is ",(x,y),", normalized ", (x1,y1))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)
            rects.append((coord[0][0], coord[0][1], coord[1][0], coord[1][1]))
            draw_count += 1
            coord = []

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()
            # remove all drawn rectangle vertices
            for i in range(draw_count):
                rects.pop(-1)

        elif key == ord("c"): # Hit 'c' to confirm the selection
            isExist = os.path.exists('gt_bboxes_img')
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs('gt_bboxes_img')
            cv2.imwrite(os.path.join('gt_bboxes_img',fname+'_'+str(page_id)+'_gt_bboxes.jpg'), clone)

            break
    # close the window
    cv2.destroyWindow(win_name)

    return rects

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # ## Required parameters
    # parser.add_argument(
    #     "--img_dir",
    #     default='../../img',
    #     type=str,
    #     required=False,
    #     help="The input data dir. Should contain the pdf files.",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     default='../../data/gt_bboxes/',
    #     type=str,
    #     required=False,
    #     help="The output directory where the output data will be written.",
    # )
    # args = parser.parse_args()

    # Prepare an image for testing
    imgdir = '../../img'
    imgfiles = sorted(glob.glob(imgdir+"*_ori.jpg"), key=lambda tup: (tup.split('/')[-1].split('_')[0],int(re.findall(r'(\d+)_ori.jpg',(tup))[0])))
    gt_bbox = {}
    # Points of the target window
    for i, imgfile in enumerate(imgfiles):
        fname1 = os.path.basename(imgfile)
        fname, page_id = re.findall(r'(.*)_(\d+)'+'_ori.jpg',fname1)[0]
        page_id = int(page_id)+1
        try:
            next_fname = os.path.basename(imgfiles[i+1])
            next_fname = re.findall(r'(.*)_(\d+)' + '_ori.jpg', next_fname)[0][0]
        except:
            pass

        if fname not in gt_bbox:
            gt_bbox[fname] = {}

        img = cv2.imread(imgfile)  # A image array with RGB color channels
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert RGB to BGR
        points = define_rect(img, fname, page_id)
        gt_bbox[fname][page_id] = points
        print(gt_bbox)
        isExist = os.path.exists('gt_bboxes')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs('gt_bboxes')
        if next_fname != fname or i==len(imgfiles)-1:
            with open(os.path.join('gt_bboxes', fname + '_gt_bboxes.docparse_json'),'w',encoding='utf8') as fp:
                fp.write(json.dumps(gt_bbox, indent=4, ensure_ascii=False))

    print("--- End ---")