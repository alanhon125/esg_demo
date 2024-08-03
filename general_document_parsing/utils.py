import random
import time
import os
import pdf2image
import pdfplumber
try:
    from api.config import *
except:
    import config

def romanToInt(s):
    """
    :type s: str
    :rtype: int
    """
    s = s.upper()
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000, 'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90,
             'CD': 400, 'CM': 900}
    i = 0
    num = 0
    try:
        while i < len(s):
            if i + 1 < len(s) and s[i:i + 2] in roman:
                num += roman[s[i:i + 2]]
                i += 2
            else:
                # print(i)
                num += roman[s[i]]
                i += 1
        return num
    except:
        return None


def isDoubleColumnDoc(pdf_path, page_id):
    '''
    vertical_strategy = text to detect tables and mess around with the text_tolerance value.
    It seems the single column begins to fail detection when set to 12
    '''
    doc = pdfplumber.open(pdf_path)
    return bool(doc.pages[page_id].extract_table(dict(vertical_strategy='text', text_tolerance=12)))


def normalize_bbox(bbox, width, height):
    '''Given actual bbox, page height and width, return normalized bbox coordinate (0-1000) on the page
    @param bbox: actual bbox in [x0,y0,x1,y1]
    @type bbox: list
    @param width: width of the page
    @type width: int
    @param height: height of the page
    @type height: int
    @rtype: [x0,y0,x1,y1], list
    @return: integral coordinates range from 0 to 1000 of a normalized bounding box
    '''
    word_bbox = (float(bbox[0]), float(bbox[1]),
                 float(bbox[2]), float(bbox[3]))
    return [min(1000, max(0, int(word_bbox[0] / width * 1000))),
            min(1000, max(0, int(word_bbox[1] / height * 1000))),
            min(1000, max(0, int(word_bbox[2] / width * 1000))),
            min(1000, max(0, int(word_bbox[3] / height * 1000)))]


def denormalize_bbox(norm_bbox, width, height):
    '''Given normalized bbox, page height and width, return actual bbox coordinate on the page
    @param norm_bbox: normalized bbox in [x0,y0,x1,y1]
    @type norm_bbox: list
    @param width: width of the page
    @type width: int
    @param height: height of the page
    @type height: int
    @return: integral coordinates of a bounding box [x0,y0,x1,y1]
    @rtype: list
    '''
    norm_bbox = (float(norm_bbox[0]), float(
        norm_bbox[1]), float(norm_bbox[2]), float(norm_bbox[3]))
    return [int(norm_bbox[0] * width / 1000),
            int(norm_bbox[1] * height / 1000),
            int(norm_bbox[2] * width / 1000),
            int(norm_bbox[3] * height / 1000)]


def denormalize_pts(norm_pts, width, height):
    norm_pts = (float(norm_pts[0]), float(norm_pts[1]))
    return [int(norm_pts[0] * width / 1000),
            int(norm_pts[1] * height / 1000)]


def bboxes_cluster_outline(bboxes_list):
    '''
    Given a list of bounding boxes (x0, y0, x1, y1),
    return a rectilinear outline bounding box that includes all bounding boxes
    @param bboxes_list: a list of bounding boxes coordinates (x0, y0, x1, y1)
    @type bboxes_list: list
    @rtype: tuple
    @return: tuple of bounding box coordinate (x0, y0, x1, y1))
    '''
    all_x0, all_y0, all_x1, all_y1 = list(zip(*bboxes_list))
    return (min(all_x0), min(all_y0), max(all_x1), max(all_y1))


def bbox_centre(bbox):
    '''
    calculate centre point of a bounding box
    @param bbox: a bounding box coordinate (x0,y0,x1,y1)
    @type bbox: list
    @rtype: tuple
    @return: tuple of coordinate (x_center, y_center)
    '''
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    return (int(x0 + w / 2), int(y0 + h / 2))


def token2bbox(wordlist, span_token, span_bbox, line_no, tol=5):
    '''
    Get bounding box by a span of tokens
    @param wordlist: word list represents in ((word_x0, word_y0, word_x1, word_y1), word_token, word_block_id, word_line_id, word_id)
    @type wordlist: list
    @param span_token: string of span token
    @type span_token: str
    @param span_bbox: bounding box of span
    @type span_bbox: list
    @param line_no: line no of the span
    @type line_no: int
    @return wbbox: bounding box of token that matches the span
    @rtype wbbox: list
    @return wordlist: updated word list after removal of matched token
    @rtype wordlist: list
    '''
    for l in wordlist:
        wx0, wy0, wx1, wy1, word_token, word_block_no, word_line_no, word_no = l
        wbbox = (wx0, wy0, wx1, wy1)
        sx0, sy0, sx1, sy1 = span_bbox
        if str(span_token.strip()) == str(word_token.strip()) and line_no == word_line_no and (int(wx0) - tol <= int(sx0) <= int(wx0) + tol or
                                                                                               int(wy0) - tol <= int(sy0) <= int(wy0) + tol or
                                                                                               int(wx1) - tol <= int(sx1) <= int(wx1) + tol or
                                                                                               int(wy1) - tol <= int(sy1) <= int(wy1) + tol):
            wordlist.remove(l)
            return wbbox, wordlist
    return None, wordlist


def neighborhood(iterable):
    iterator = iter(iterable)
    prev_item = None
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (prev_item, current_item, next_item)
        prev_item = current_item
        current_item = next_item
    yield (prev_item, current_item, None)


def min_offset(tgt_bbox, ref_bbox):
    '''
    Function to obtain the min. x, y coordinate offset between two bounding boxes
    '''
    return (
        min(abs(tgt_bbox[i] - ref_bbox[j])
            for i, j in ((0, 2), (0, 0), (2, 0), (2, 2))),
        min(abs(tgt_bbox[i] - ref_bbox[j])
            for i, j in ((1, 3), (1, 1), (3, 1), (3, 3)))
    )


def is_bbox_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap.

    Args:
        bbox1: A tuple of four values (x0, y0, x1, y1) representing the first bounding box.
        bbox2: A tuple of four values (x0, y0, x1, y1) representing the second bounding box.

    Returns:
        A boolean indicating whether the two bounding boxes overlap.
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # Check if there is no overlap
    if x0_1 > x1_2 or x0_2 > x1_1 or y0_1 > y1_2 or y0_2 > y1_1:
        return False

    # Otherwise, there is overlap
    return True


def is_adjacent_bbox(tgt_bbox, ref_bbox, xoffset_thres=1000, yoffset_thres=1000):
    '''
    Function to check if the target bounding box is adjacent to the reference bounding box given the tolerance of x- and y-coordinate offsets
    @param tgt_bbox: target bounding box in (x0,y0,x1,y1)
    @type tgt_bbox: list
    @param ref_bbox: reference bounding box in (x0,y0,x1,y1)
    @type ref_bbox: list
    @param xoffset: x-coordinate tolerance to accept bounding box as adjacent bbox
    @type xoffset: int or float
    @param yoffset: y-coordinate tolerance to accept bounding box as adjacent bbox
    @type yoffset: int or float
    @return: boolean variable indicate if the target bbox is adjacent to reference bbox
    @rtype: bool
    '''
    x_offset, y_offset = min_offset(tgt_bbox, ref_bbox)

    if x_offset <= xoffset_thres and y_offset <= yoffset_thres:
        return True
    else:
        return False


def find_adjacent_bbox(tgt_bbox, ref_bboxes, xoffset=(20, 50), yoffset=(10, 30)):
    '''
    Function to list out all bounding boxes in reference bounding boxes list that adjacent to the target bounding boxes
    @param tgt_bbox: target bounding box in (x0,y0,x1,y1)
    @type tgt_bbox: list
    @param ref_bboxes: list of reference bounding box in (x0,y0,x1,y1)
    @type ref_bboxes: list or tuple
    @param xoffset: tuple of x-coordinate min, max tolerance to accept bounding box as adjacent bbox (default = page width /3)
    @type xoffset: tuple
    @param yoffset: tuple of y-coordinate min, max tolerance to accept bounding box as adjacent bbox (default = 30)
    @type yoffset: tuple
    @return: list of tuple of (adjacent bounding box in (x0,y0,x1,y1), adjacency direction), adjacency direction "0" as horizontal, "1" as vertical
    @rtype: list
    '''
    adjacent_bboxes = {'left': [],
                       'right': [],
                       'up': [],
                       'down': [],
                       'upper-left': [],
                       'upper-right': [],
                       'lower-left': [],
                       'lower-right': []}
    tgt_x0, tgt_y0, tgt_x1, tgt_y1 = tgt_bbox

    for ref_bbox in ref_bboxes:
        x0, y0, x1, y1 = ref_bbox
        if tgt_bbox == ref_bbox:
            continue
        tgt_left_vs_right = abs(tgt_x0 - x1)
        tgt_right_vs_left = abs(tgt_x1 - x0)
        tgt_left_vs_left = abs(tgt_x0 - x0)
        tgt_top_vs_top = abs(tgt_y0 - y0)
        tgt_top_vs_btm = abs(tgt_y0 - y1)
        tgt_btm_vs_top = abs(tgt_y1 - y0)
        if tgt_left_vs_left <= xoffset[0]:
            if tgt_top_vs_btm <= yoffset[1]:
                adjacent_bboxes['up'].append(ref_bbox)
            elif tgt_btm_vs_top <= yoffset[1]:
                adjacent_bboxes['down'].append(ref_bbox)
        elif tgt_top_vs_top <= yoffset[0]:
            if tgt_right_vs_left <= xoffset[1]:
                adjacent_bboxes['right'].append(ref_bbox)
            elif tgt_left_vs_right <= xoffset[1]:
                adjacent_bboxes['left'].append(ref_bbox)
        elif tgt_left_vs_right <= xoffset[0]:
            if tgt_top_vs_btm <= yoffset[1]:
                adjacent_bboxes['upper-left'].append(ref_bbox)
            elif tgt_btm_vs_top <= yoffset[1]:
                adjacent_bboxes['lower-left'].append(ref_bbox)
        elif tgt_right_vs_left <= xoffset[0]:
            if tgt_top_vs_btm <= yoffset[1]:
                adjacent_bboxes['upper-right'].append(ref_bbox)
            elif tgt_btm_vs_top <= yoffset[1]:
                adjacent_bboxes['lower-right'].append(ref_bbox)
    return adjacent_bboxes

# get the mode of tags in surrounding texts


def surround_common(token_tag_dict, bbox, page_id, tgt_tag, exclude_tags, xoffset_thres=1000, yoffset_thres=10):
    '''
    Get the most common tag surrounded. If there is no text surrounding, return original tag
    @param token_tag_dict: the dictionary of token info, including (token, bounding box [x0,y0,x1,y1], page id, rule-based tag of token, model prediction tag of token, hybrid prediction tag of token)
    @type token_tag_dict: dict
    @param bbox: the text bounding box of the target
    @type bbox: list
    @param page_id: the page id of the target (first page id as 1)
    @type page_id: int
    @param tgt_tag: the element tag of the target
    @type tgt_tag: str
    @param exclude_tags: the element tag that should exclude as surrounding tag
    @type exclude_tags: list
    @param xoffset: x-coordinate tolerance to accept bounding box as surrounding bbox (default = 1000)
    @type xoffset: int
    @param yoffset: y-coordinate tolerance to accept bounding box as surrounding bbox (default = 10)
    @type yoffset: int
    @return: the most common tag surrounded
    @rtype: str
    '''
    tags = [
        item['hybrid_tag']
        for item in token_tag_dict
        if item['page_id'] == page_id
        and (xoffset := min_offset(bbox, item['bbox'])[0]) <= xoffset_thres
        and (yoffset := min_offset(bbox, item['bbox'])[1]) <= yoffset_thres
        and item['hybrid_tag'] not in exclude_tags
    ]

    return max(set(tags), key=tags.count) if tags else tgt_tag


def page_size(doc, page_id):
    '''Extract page weight and height with given page id
    @param doc: PyMuPDF class that represent the document
    @type doc: <class 'fitz.fitz.Document'>
    @param page_id: page id (first page id as 0)
    @type page_id: int
    @return w,h : weight, height of the page
    @rtype w,h : tuple
    '''
    w, h = int(doc[page_id].rect.width), int(doc[page_id].rect.height)
    return w, h


def create_save_pdf_img(pdf_inpath, img_outdir, fname):
    '''
    create pdf page images by pdf2image libary and output to image folder
    @param pdf_inpath: path to pdf
    @type pdf_inpath: str
    @param img_outdir: the output directory where the output page images will be written
    @type img_outdir: str
    @param fname: filename
    @type fname: str
    '''

    from pdf2image import pdfinfo_from_path, convert_from_path
    info = pdfinfo_from_path(pdf_inpath, userpw=None, poppler_path=None)

    maxPages = info["Pages"]
    all_img_paths = [os.path.join(
        img_outdir, fname + f'_{str(i)}_ori.jpg') for i in range(1, maxPages + 1)]
    check_existence = [os.path.exists(img_path) for img_path in all_img_paths]
    if all(check_existence):
        return

    chunkPages = 100
    count = 0
    for page in range(1, maxPages + 1, chunkPages):
        pdf_images = convert_from_path(
            pdf_inpath, dpi=200, first_page=page, last_page=min(page + chunkPages - 1, maxPages))
        for page_id in range(chunkPages):  # save document page images
            if count > maxPages-1:
                break
            pdf_images[page_id].save(os.path.join(
                img_outdir, fname + '_{}_ori.jpg'.format(str(page_id + page))))
            count += 1


def annot_pdf_page(doc, page_id, text_label, norm_bbox, color=None):
    '''
    Annotate the page with given page id, label and bounding box
    @param doc: PyMuPDF class that represent the document
    @type doc: <class 'fitz.fitz.Document'>
    @param page_id:page id
    @type page_id: int
    @param text_label: text label above bounding box on an annotated page
    @type text_label: str
    @param norm_bbox: normalized bounding box in (x0, y0, x1, y1)
    @type norm_bbox: list
    @param color: list
    @type color: list
    @return: annotated document page
    @rtype: <class 'fitz.fitz.Document'>
    '''
    if color is None:
        random.seed(time.process_time())
        color = (random.random(), random.random(), random.random())
    w, h = page_size(doc, page_id)
    doc[page_id].clean_contents()
    x0, y0, x1, y1 = denormalize_bbox(norm_bbox, w, h)
    # pts = [self.denormalize_pts(pt, w, h) for pt in norm_bbox]
    # doc[page_id].insert_text((pts[0][0], pts[0][1] - 2), text_label, fontsize=8, color=color)
    # doc[page_id].draw_polyline(pts, color=color, width=1)
    doc[page_id].insert_text((x0, y0 - 2), text_label, fontsize=8, color=color)
    doc[page_id].draw_rect((x0, y0, x1, y1), color=color, width=1)
    return doc


def annot_pdf_page_polygon(doc, page_id, text_label, points, color=None):
    '''
    Annotate the page with given page id, label and bounding box
    @param doc: PyMuPDF class that represent the document
    @type doc: <class 'fitz.fitz.Document'>
    @param page_id:page id
    @type page_id: int
    @param text_label: text label above bounding box on an annotated page
    @type text_label: str
    @param points: normalized points of vertice in (x,y)
    @type points: list
    @param color: list
    @type color: list
    @return: annotated document page
    @rtype: <class 'fitz.fitz.Document'>
    '''
    if color is None:
        random.seed(time.process_time())
        color = (random.random(), random.random(), random.random())
    w, h = page_size(doc, page_id)
    doc[page_id].clean_contents()
    points = [(int(x * w / 1000), int(y * h / 1000)) for x, y in points]
    upper_left_pt = min(points)
    doc[page_id].insert_text(
        (upper_left_pt[0], upper_left_pt[1] - 2), text_label, fontsize=8, color=color)
    doc[page_id].draw_polyline(points, color=color, width=1)

    return doc


def create_folder(out_folders):
    '''Create folder for data output if the directory doesn't exist'''
    if isinstance(out_folders, list):
        for f in out_folders:
            isExist = os.path.exists(f)
            if not isExist:
                os.makedirs(f)
    else:
        isExist = os.path.exists(out_folders)
        if not isExist:
            os.makedirs(out_folders)


def sub_illegal_char(txt):
    '''
    substitute illegal character with empty string
    @param txt: input character
    @dtype txt: string
    @return: string of replaced text
    '''
    import re

    def isfloat(num):
        '''
        Check if the input is a float/integer, return True if it is otherwise False
        @param num: input in any data type
        @return: boolean
        '''
        if num in [None, True, False]:
            return False
        try:
            float(num)
            return True
        except ValueError:
            return False

    ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    if isfloat(txt):
        return float(txt)
    else:
        txt = ILLEGAL_CHARACTERS_RE.sub(r'', str(txt))
    return txt


def multiprocess(function, input_list, args=None):
    '''
    multiprocessing the function at a time

    @param function: a function
    @type function: def
    @param input_list: a list of input that accept by the function
    @type input_list: list
    @param args: arguments variables
    @type args: any argument type as required by function
    '''
    import multiprocessing
    
    def contains_explicit_return(function):
        # Check if function has return statement
        import ast
        import inspect
        
        return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(function))))

    input_length = len(input_list)
    num_processor = multiprocessing.cpu_count()
    print(f'there are {num_processor} CPU cores')
    batch_size = max(input_length // num_processor, 1)
    num_batch = int(input_length / batch_size) + (input_length % batch_size > 0)
    pool = multiprocessing.Pool(num_processor)
    if not args:
        processes = [pool.apply_async(function, args=(input_list[idx * batch_size:(idx + 1) * batch_size],)) for idx in range(num_batch)]
    else:
        processes = [pool.apply_async(function, args=(input_list[idx * batch_size:(idx + 1) * batch_size], args)) for idx in range(num_batch)]
    
    if contains_explicit_return(function): # if function has return statement
        results = [p.get() for p in processes]
        results = [i for r in results if isinstance(r,list) or isinstance(r,tuple) for i in r]
        
        return results
    else:
        for p in processes:
            p.get()


def get_polygon(rectangles):
    """
    Given a list of non-overlapping rectangle coordinates in tuples of (x0,y0,x0,y0),
    return the tuples of coordinates of vertices (x0,x0,y0,y0) forming the outlines of the final rectilinear polygon,
    sorting from upper bottom-right vertice and in anti-clockwise direction.

    e.g. rectangles = [(0,0,1,1),(1,0,2,1),(2,0,3,1),(3,0,4,1),(0,1,1,2),(1,1,2,2),(2,1,3,2)]
    return polygons = [[(3, 2), (3, 1), (4, 1), (4, 0), (0, 0), (0, 2)]]

    Algorithm:
        1. Sort points by lowest x, lowest y
        2. Go through each column and create edges between the vertices 2i and 2i + 1 in that column
        3. Sort points by lowest y, lowest x
        4. Go through each row and create edges between the vertices 2i and 2i + 1 in that row.
    """
    points = set()
    for (x0, y0, x1, y1) in rectangles:
        for pt in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            if pt in points:  # Shared vertice, remove it.
                points.remove(pt)
            else:
                points.add(pt)
    points = list(points)

    def y_then_x(a, b):
        if a[1] < b[1] or (a[1] == b[1] and a[0] < b[0]):
            return -1
        elif a == b:
            return 0
        else:
            return 1

    def cmp_to_key(mycmp):
        'Convert a cmp= function into a key= function'

        class K(object):
            def __init__(self, obj, *args):
                self.obj = obj

            def __lt__(self, other):
                return mycmp(self.obj, other.obj) < 0

            def __gt__(self, other):
                return mycmp(self.obj, other.obj) > 0

            def __eq__(self, other):
                return mycmp(self.obj, other.obj) == 0

            def __le__(self, other):
                return mycmp(self.obj, other.obj) <= 0

            def __ge__(self, other):
                return mycmp(self.obj, other.obj) >= 0

            def __ne__(self, other):
                return mycmp(self.obj, other.obj) != 0

        return K

    sort_x = sorted(points)
    sort_y = sorted(points, key=cmp_to_key(y_then_x))

    edges_h = {}
    edges_v = {}

    i = 0
    while i < len(points):
        curr_y = sort_y[i][1]
        while i < len(points) and sort_y[i][1] == curr_y:
            edges_h[sort_y[i]] = sort_y[i + 1]
            edges_h[sort_y[i + 1]] = sort_y[i]
            i += 2

    i = 0
    while i < len(points):
        curr_x = sort_x[i][0]
        while i < len(points) and sort_x[i][0] == curr_x:
            edges_v[sort_x[i]] = sort_x[i + 1]
            edges_v[sort_x[i + 1]] = sort_x[i]
            i += 2

    # Get all the polygons.
    p = []
    while edges_h:
        # We can start with any point.
        polygon = [(edges_h.popitem()[0], 0)]
        while True:
            curr, e = polygon[-1]
            if e == 0:
                next_vertex = edges_v.pop(curr)
                polygon.append((next_vertex, 1))
            else:
                next_vertex = edges_h.pop(curr)
                polygon.append((next_vertex, 0))
            if polygon[-1] == polygon[0]:
                # Closed polygon
                # polygon.pop()
                break
        # Remove implementation-markers from the polygon.
        poly = [point for point, _ in polygon]
        for vertex in poly:
            if vertex in edges_h:
                edges_h.pop(vertex)
            if vertex in edges_v:
                edges_v.pop(vertex)

        p.append(poly)

        if len(p) == 1:
            return p[0]
        else:
            return p

def list2dict_2(text, nlp, use_nested=True, override_key=None, delimiter_pattern=PARA_SPLIT_PATTERN):
    import re
    import string

    bullet_list_symbol = '|'.join(SYMBOLS)

    # @timeit
    def spliting_text(text):

        # numbering pattern in brackets should not precede by ), non-whitespace character and not succeed by (
        brackets_pattern = '(?<!\), )(?<!\))(?<![^\s\[])\(' + '\)(?!\()' \
                                         '|' \
                                         '(?<!\), )(?<!\))(?<![^\s\[])\('.join(NUMBERING_LIST) + '\)(?!\()'
        # numbering pattern with right bracket only should not succeed by (
        right_bracket_pattern = '(?<!\), )(?<!\()(?<![^\s\[])' + '\)(?!\()' \
                                            '|' \
                                            '(?<!\), )(?<!\()(?<![^\s\[])'.join(NUMBERING_LIST) + '\)(?!\()'
        # numbering pattern with dot should not succeed by any digit
        dot_pattern = '(?<![a-zA-Z]\.)' + '\.(?!\d)(?![a-zA-Z]\.)|(?<![a-zA-Z]\.)'.join(
            NUMBERING_LIST) + '\.(?!\d)(?![a-zA-Z]\.)'
        pattern = '(' + brackets_pattern + '|' + \
            right_bracket_pattern + '|' + dot_pattern + ')'

        splits = [i.strip() for i in re.split(
            pattern, text) if i and i.strip()]
        splits = [x for x in splits if not re.match(
            rf'^{ROMAN_NUM}$', str(x), flags=re.IGNORECASE)]
        splits = [x for x in splits if x not in string.punctuation]

        if len(splits) <= 1:
            return splits

        # print(json.dumps(splits, indent=4))
        def insert_numbering_at_even(curr_txt, split_list):
            if re.match(pattern, curr_txt) and len(split_list) % 2 == 0:
                split_list.append(curr_txt)
            elif not re.match(pattern, curr_txt) and len(split_list) % 2 == 1:
                split_list.append(curr_txt)
            else:
                if len(split_list) > 0 and not re.match(pattern, split_list[-1]):
                    last = split_list.pop(-1)
                    split_list.append(last + ' ' + curr_txt)
                else:
                    split_list.append(' ')
                    split_list.append(curr_txt)
            return split_list

        iterator = neighborhood(splits)
        tmp_splits = []
        continue_again = False
        for prev, curr, nxt in iterator:
            if continue_again:
                continue_again = False
                continue
            if not prev:
                tmp_splits = insert_numbering_at_even(curr, tmp_splits)
                continue
            else:
                prev = tmp_splits[-1]
                if re.match(pattern, curr) and ((curr.endswith(')') and re.search(r'.*' + rf'{r"$|.*".join(NLB_BRACK)}' + r'$', prev)) or ((curr.endswith('.') and re.search(r'.*' + rf'{r"$|.*".join(NLB_DOT)}' + r'$', prev)))):
                    prev = tmp_splits.pop(-1)
                    if not prev.endswith('.'):
                        prev += ' '
                    if not curr.endswith('.'):
                        curr += ' '
                    if nxt:
                        tmp_splits = insert_numbering_at_even(
                            prev + curr + nxt, tmp_splits)
                    else:
                        tmp_splits = insert_numbering_at_even(
                            prev + curr, tmp_splits)
                    continue_again = True
                    continue
                # elif re.match(pattern, prev) and re.match(pattern, curr):
                #     tmp_splits.append(' ')
                #     tmp_splits.append(curr)
                else:
                    tmp_splits = insert_numbering_at_even(curr, tmp_splits)
            if nxt:
                if re.match(pattern, curr) and ((curr.endswith(')') and re.search(r'^' + rf'{r".*|^".join(NLA_NUM_BRACK)}' + r'.*', nxt)) or ((curr.endswith('.') and re.search(r'^' + rf'{r".*|^".join(NLA_NUM_DOT)}' + r'.*', nxt)))):
                    curr = tmp_splits.pop(-1)
                    if not curr.endswith('.'):
                        curr += ' '
                    tmp_splits = insert_numbering_at_even(
                        curr + nxt, tmp_splits)
                    continue_again = True
                    continue

        # print(json.dumps(tmp_splits,indent=4))

        return tmp_splits

    splits = spliting_text(text)

    # @timeit
    def split_bullet(text):
        # split string into list with bullet point characters
        # if len(splits) <= 1:
        splits = re.split(bullet_list_symbol, text)
        splits = [x.strip() for x in splits if x != '' and x != ' ']
        if len(splits) > 1:
            return splits
        else:
            return phrase_tokenize(text, nlp, delimiter_pattern=delimiter_pattern)

    if len(splits) <= 1:
        splits = split_bullet(text)
        return splits

    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    splits = pairwise(splits)

    # @timeit
    def hierarchical_numbering(splits):
        dic = {}
        primaryLevel = secondaryLevel = tertiaryLevel = quaternaryLevel = quinaryLevel = ''
        last_primary_key = last_secondary_key = last_tertiary_key = last_quaternary_key = last_quinary_key = ''
        last_pattern = ''
        if override_key:
            keys = re.findall(r'(\(*\w+\)*\.*)', override_key)
            if len(keys) <= 4:
                i = 1
                while keys:
                    if i == 1:
                        last_primary_key = keys.pop(0)
                        for p in NUMBERING_LIST:
                            if re.match(p, last_primary_key.strip("().")):
                                primaryLevel = p
                                break
                    elif i == 2:
                        last_secondary_key = keys.pop(0)
                        for p in NUMBERING_LIST:
                            if re.match(p, last_secondary_key.strip("().")):
                                secondaryLevel = p
                                break
                    elif i == 3:
                        last_tertiary_key = keys.pop(0)
                        for p in NUMBERING_LIST:
                            if re.match(p, last_tertiary_key.strip("().")):
                                tertiaryLevel = p
                                break
                    elif i == 4:
                        last_quaternary_key = keys.pop(0)
                        for p in NUMBERING_LIST:
                            if re.match(p, last_quaternary_key.strip("().")):
                                quaternaryLevel = p
                                break
                    i += 1

        for prev, curr, nxt in neighborhood(list(splits)):
            curr_key, curr_value = curr
            curr_value = tuple(re.split(bullet_list_symbol, curr_value))
            if len(curr_value) == 1:
                curr_value = curr_value[0]
            c_key = curr_key.strip("().")
            if prev is None and override_key is None:
                for p in NUMBERING_LIST:
                    if re.match(p, c_key):
                        primaryLevel = p
                        break
                p_key = None
            elif prev is None and override_key:
                p_key = last_primary_key.strip("().")
                for p in NUMBERING_LIST:
                    if re.match(p, c_key):
                        secondaryLevel = p
                        break
            else:
                prev_key, prev_value = prev
                p_key = prev_key.strip("().")
            if nxt is not None:
                next_key, next_value = nxt
                n_key = next_key.strip("().")
            else:
                n_key = None

            def isNextNum(prev_num, curr_num):
                # if previous number is None or empty string
                if prev_num is None or prev_num == '':
                    return True
                else:
                    prev_num = prev_num.strip('()')
                # if two numbers are digits
                if str(curr_num).isdigit() and str(prev_num).isdigit():
                    if int(curr_num) == int(prev_num) + 1:
                        return True
                try:
                    # if two numbers are alphabet character
                    if curr_num == chr(ord(prev_num) + 1):
                        return True
                    else:
                        try:
                            prev_num = romanToInt(prev_num)
                            curr_num = romanToInt(curr_num)
                            # if two numbers are roman numbers character
                            if prev_num and curr_num:
                                if curr_num == prev_num + 1:
                                    return True
                        except:
                            return False
                except:
                    return False

            if re.match(primaryLevel, c_key):
                last_primary_key = curr_key
                last_primary_value = curr_value
            elif re.match(secondaryLevel, c_key):
                last_secondary_key = curr_key
                last_secondary_value = curr_value
            elif re.match(tertiaryLevel, c_key):
                last_tertiary_key = curr_key
                last_tertiary_value = curr_value
            elif re.match(quaternaryLevel, c_key):
                last_quaternary_key = curr_key
                last_quaternary_value = curr_value

            for i, k in enumerate(NUMBERING_LIST):
                if n_key is not None:
                    match_c = re.match(k, c_key)
                    match_n = re.match(k, n_key)
                    if match_c and match_n:
                        match_n = isNextNum(c_key, n_key)
                    not_same_pattern = (match_c and (
                        not match_n or match_n is None))
                    if not_same_pattern or match_n == False:
                        if k == primaryLevel:
                            for p in NUMBERING_LIST:
                                if p == primaryLevel:
                                    continue
                                if re.match(p, n_key):
                                    secondaryLevel = p
                                    break
                            tmp = {curr_key: {curr_value: {}}}
                        elif k == secondaryLevel:
                            for p in NUMBERING_LIST:
                                if p == secondaryLevel:
                                    continue
                                if re.match(p, n_key):
                                    tertiaryLevel = p
                                    break
                            if re.match(primaryLevel, n_key):
                                tmp = {curr_key: curr_value}
                            else:
                                tmp = {curr_key: {curr_value: {}}}
                        elif k == tertiaryLevel:
                            for p in NUMBERING_LIST:
                                if p == tertiaryLevel:
                                    continue
                                if re.match(p, n_key):
                                    quaternaryLevel = p
                                    break
                            if re.match(primaryLevel, n_key) or re.match(secondaryLevel, n_key):
                                tmp = {curr_key: curr_value}
                            else:
                                tmp = {curr_key: {curr_value: {}}}
                        elif k == quaternaryLevel:
                            for p in NUMBERING_LIST:
                                if p == quaternaryLevel:
                                    continue
                                if re.match(p, n_key):
                                    quinaryLevel = p
                                    break
                            if re.match(primaryLevel, n_key) or re.match(secondaryLevel, n_key) or re.match(
                                    tertiaryLevel, n_key):
                                tmp = {curr_key: curr_value}
                            else:
                                tmp = {curr_key: {curr_value: {}}}
                        else:
                            tmp = {curr_key: curr_value}
                        break
                else:
                    tmp = {curr_key: curr_value}
                    break
                if i == len(NUMBERING_LIST) - 1:
                    tmp = {curr_key: curr_value}

            if re.match(primaryLevel, c_key) or (last_pattern == primaryLevel and isNextNum(p_key, c_key)):
                last_pattern = primaryLevel
                if use_nested:
                    dic.update(tmp)
                else:
                    if curr_key in dic.keys():
                        curr_key += ' '
                    dic.update({curr_key: curr_value})
            elif re.match(secondaryLevel, c_key) or (last_pattern == secondaryLevel and isNextNum(p_key, c_key)):
                last_pattern = secondaryLevel
                if use_nested:
                    dic[last_primary_key][last_primary_value].update(tmp)
                else:
                    if last_primary_key + curr_key in dic.keys():
                        curr_key += ' '
                    dic.update({last_primary_key + curr_key: curr_value})
            elif re.match(tertiaryLevel, c_key) or (last_pattern == tertiaryLevel and isNextNum(p_key, c_key)):
                last_pattern = tertiaryLevel
                if use_nested:
                    dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value].update(
                        tmp)
                else:
                    if last_primary_key + last_secondary_key + curr_key in dic.keys():
                        curr_key += ' '
                    dic.update(
                        {last_primary_key + last_secondary_key + curr_key: curr_value})
            elif re.match(quaternaryLevel, c_key) or (last_pattern == quaternaryLevel and isNextNum(p_key, c_key)):
                last_pattern = quaternaryLevel
                if use_nested:
                    dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value][
                        last_tertiary_key][last_tertiary_value].update(tmp)
                else:
                    if last_primary_key + last_secondary_key + last_tertiary_key + curr_key in dic.keys():
                        curr_key += ' '
                    dic.update({last_primary_key + last_secondary_key +
                               last_tertiary_key + curr_key: curr_value})
            elif re.match(quinaryLevel, c_key) or (last_pattern == quinaryLevel and isNextNum(p_key, c_key)):
                last_pattern = quinaryLevel
                if use_nested:
                    dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value][
                        last_tertiary_key][
                        last_tertiary_value][last_quaternary_key][last_quaternary_value].update(tmp)
                else:
                    if last_primary_key + last_secondary_key + last_tertiary_key + last_quaternary_key + curr_key in dic.keys():
                        curr_key += ' '
                    dic.update({
                        last_primary_key + last_secondary_key + last_tertiary_key + last_quaternary_key + curr_key: curr_value})
        return dic

    dic = hierarchical_numbering(splits)
    return dic

def list2dict(text, nlp, use_nested=True, override_key=None):
    import re
    import inflect
    import string

    xyz = r'\b[xyz]\b'
    capital_xyz = r'\b[XYZ]\b'
    capital_roman_number_pattern = r'\b(?=[XVI])M*(X[L]|L?X{0,2})(I[XV]|V?I{0,3})\b'
    roman_number_pattern = r'\b(?=[xvi])m*(x[l]|l?x{0,2})(i[xv]|v?i{0,3})\b'
    capital_alphabet_pattern = r'\b[^\d\sIVXivxa-zXYZ\W_]{1}\b'
    alphabet_pattern = r'\b[^\d\sIVXivxA-Zxyz\W_]{1}\b'
    digit_pattern = r'\b[1-9]\d{0,1}\b'
    multi_dot_digit_pattern1 = rf'{digit_pattern}\.{digit_pattern}\.{digit_pattern}'
    multi_dot_digit_pattern2 = rf'{digit_pattern}\.{digit_pattern}'
    numbering_patterns = [xyz, capital_xyz, capital_roman_number_pattern, roman_number_pattern,
                          capital_alphabet_pattern, alphabet_pattern, multi_dot_digit_pattern1, multi_dot_digit_pattern2, digit_pattern]
    alphabets = ['[a-z]', '[a-z]{2}', '[a-z]{3}', '[a-z]{4}', '[a-z]{5}', '[A-Z]', '[A-Z]{2}', '[A-Z]{3}', '[A-Z]{4}', '[A-Z]{5}']
    numWithBrack1 = ['\(' + i + '\)' for i in alphabets]
    numWithBrack2 = ['\(' + i + '\) and' for i in alphabets]
    numWithBrack3 = ['\(' + i + '\),' for i in alphabets]
    numWithBrack4 = ['\(' + i + '\) to' for i in alphabets]
    numWithBrack5 = ['\(' + i + '\) or' for i in alphabets]
    numWithBrack = numWithBrack2 + numWithBrack3 + numWithBrack4 + numWithBrack5 # numWithBrack1 +
    p = inflect.engine()
    numInWord = [p.number_to_words(i).capitalize() for i in range(100)] + [p.number_to_words(i) for i in range(100)]
    PREPOSITIONS = ['\babout\b', '\babove\b', '\bacross\b', '\band\b', '\bat\b', '\bbelow\b', '\bbesides\b', '\bby\b', '\bfor\b', '\bfrom\b', '\bin\b', '\bincrease\b', '\bof\b', '\bon\b', '\bor\b', '\breduce\b', '\bthan\b', '\bto\b', '\bunder\b', '\bwith\b', '\bwithin\b', ' to ', '\bafter\b', 'and\\\or']
    # negative lookbehind list (list of patterns that should not be preceded by numbering) for numbering with bracket, e.g not to extract numbering with pattern Clause (a), Paragraph (e) etc.
    NLB_NUM_BRACK = ['<', '>', '\&', '\(\w[\)|\.]\s\band\b', '\w[\)|\.]\s\band\b', 'Agreement', 'agreement', 'Article', 'article', 'Articles', 'articles',
                           'Basel', 'Basel', 'BORROWER', 'Borrower', 'clause', 'Clause', 'clause\(s\)', 'clauses', 'Clauses', 'Column', 'column', 'Columns', 'columns', 'Company',
                           'company', 'Counsel', 'counsel', 'CRD', 'equal to', 'Facility', 'facility', 'General', 'general','greater than', 'less than','limbs', 'm', 'para', 'Paragraph', 'paragraph', 'Paragraph\(s\)', 'paragraph\(s\)', 
                           'Paragraphs', 'paragraphs', 'Premium', 'premium', 'Property', 'property', 'Section', 'section', 'Sections', 'sections', 'sub-paragraph', 'Sub-paragraph',
                           'sub-paragraphs', 'Sub-paragraphs', 'Tranche', 'tranche', 'Unit', 'unit', 'within']
    # negative lookbehind list (list of patterns that should not be preceded by numbering) for numbering with dot, e.g not to extract numbering with pattern Clause 1., Paragraph e. etc.
    NLB_NUM_DOT = [',', '<', '>', '\&', '\[', '\d', 'Agreement', 'agreement', 'Article', 'article', 'Articles', 'articles', 'at least', 'at most', 'Basel', 'Basel', 
                            'BORROWER', 'Borrower', 'clause', 'Clause', 'clause\(s\)', 'clauses', 'Clauses', 'Column', 'column', 'columns', 'Colums', 'Company', 'company', 'Counsel', 'counsel','CRD', 
                            'equal to', 'exceed', 'Facility', 'facility', 'General', 'general','greater than', 'less than', 'limbs', 'm', 'para', 'Paragraph', 'paragraph', 'Paragraph\(s\)', 'paragraph\(s\)', 'Paragraphs', 
                            'paragraphs', 'Premium', 'premium', 'Property', 'property', 'Section', 'section', 'Sections', 'sections', 'sub-paragraph', 'Sub-paragraph', 'sub-paragraphs', 
                            'Sub-paragraphs', 'Tranche', 'tranche', 'Unit', 'unit'] + PREPOSITIONS  # Asserts that what immediately precedes the current position in the string is not in this list
    # negative lookahead list (list of patterns that is not present immediately after numbering) for numbering with dot, e.g not to extract and split numbering with pattern 1.0, 2.1 years etc.
    NLA_NUM_DOT = [' \babove\b', '0', '[a-zA-Z]', '\d{1}', '\d{1} \%', '\d{1} \(', '\d{1} per cent', '\d{1} percent', '\d{1} years', '\d{1}\%', '\d{1}\([a-zA-Z]\)', '\d{1}x', '\d{2}', '\d{2} \%', '\d{2} \(', '\d{2} per cent', '\d{2} percent', '\d{2} years', '\d{2}\%', '\d{2}\([a-zA-Z]\)', 'm\.', 'x']  # Asserts that what immediately follows the current position in the string is not in this list
    # negative lookahead list (list of patterns that is not present immediately after numbering) for numbering with bracket, e.g not to extract and split numbering with pattern 1)., (2)& etc.
    NLA_NUM_BRACK = [re.escape(i) for i in string.punctuation]

    NLB_BRACK = NLB_NUM_BRACK + numInWord + numWithBrack
    NLB_BRACK = rf'(?<!{")(?<!".join(NLB_BRACK)})'
    NLB_DOT = NLB_NUM_DOT + numInWord + numWithBrack
    NLB_DOT = rf'(?<!{")(?<!".join(NLB_DOT)})'
    NLA_DOT = rf'(?!{")(?!".join(NLA_NUM_DOT)})'
    NLA_BRACK = rf'(?!{")(?!".join(NLA_NUM_BRACK)})'

    DELIMITERS = ['; and', '; or', 'and;', 'or;', 'to :', ';']

    not_begin_with = r'\s\w€¥' + ''.join([re.escape(i) for i in string.punctuation])
    begin_pattern = rf'(\n+|^|[^{not_begin_with}]|{NLB_BRACK}\s|' + r"\s|".join(DELIMITERS) + r'\s)'
    begin_pattern2 = rf'(\n+|^|[^{not_begin_with}]|{NLB_DOT}\s|' + r"\s|".join(DELIMITERS) + r'\s)'
    tmp = r'{2}\b\)' + f'{NLA_BRACK}|{begin_pattern}\(*'
    multi_alphabet_pattern = rf'{begin_pattern}\(*\b' + tmp.join([i for i in string.ascii_lowercase]) + r'{2}\b\)' + NLA_BRACK

    begin_pattern_with_dot = f'{begin_pattern2}' + rf'|{begin_pattern2}'.join([i + rf'\.\s*{NLA_DOT}' for i in numbering_patterns])
    begin_pattern_with_bracket = f'{begin_pattern}\(*' + rf'\){NLA_BRACK}|{begin_pattern}\(*'.join(numbering_patterns) + f'\){NLA_BRACK}' + f'|{multi_alphabet_pattern}'
    patterns_with_bracket = r'^\(*' + f'\){NLA_BRACK}|^\(*'.join(numbering_patterns) + rf'\){NLA_BRACK}'
    patterns_with_dot = r'^' + f'\.\s*{NLA_DOT}|'.join(numbering_patterns) + rf'\.\s*{NLA_DOT}'
    patterns_with_begin = '(' + begin_pattern_with_bracket + '|' + begin_pattern_with_dot + ')'
    all_pattern = '(' + patterns_with_bracket + '|' + patterns_with_dot + ')'
    startwith_patterns = r'^(\(*' + '\).*|\(*'.join(numbering_patterns) + r'\).*)$' + r'|(^' + '\..*|'.join(numbering_patterns) + r'\..*)$'
    symbol_list = ['●', '•', '·', '∙', '◉', '○', '⦿', '。', '■', '□', '☐', '⁃', '◆', '◇', '◈', '✦', '➢', '➣', '➤', '‣', '▶', '▷', '❖', '_']
    bullet_list_symbol = '|'.join(symbol_list)
    # bullet_list_symbol = '[^\S]'+'|[^\S]'.join(symbol_list)

    splits = re.split(patterns_with_begin, text) # remove split single bullet key without bracker (e.g. a, iv, 1 etc.)
    splits = [x for x in splits if not re.match(rf'^{roman_number_pattern}$|^{alphabet_pattern}$|^{digit_pattern}$', str(x), flags=re.IGNORECASE)]
    splits = [x.replace("\n", " ").strip() for x in splits if isinstance(x,str)]
    splits = [x for x in splits if x != '' and x != '\n' and x is not None]  # remove empty string and None

    all_replace = [NLB_BRACK, NLB_DOT, NLA_DOT, NLA_BRACK] + DELIMITERS
    for i in all_replace:
        # replace numbering into empty string
        splits = [re.sub(rf'^{i}$', '', x).strip() for x in splits]
        splits = [re.sub(rf"^{i}\s(\(\w+\))$", r'\1', x).strip() for x in splits]  # replace numbering with beginning pattern and brackets as number with brackets
        splits = [re.sub(rf"^{i}\s(\w+\))$", r'\1', x).strip() for x in splits]  # replace numbering with beginning pattern and right bracket as number with right bracket
        splits = [re.sub(rf"^{i}\s(\w+\.)$", r'\1', x).strip() for x in splits]  # replace numbering with beginning pattern and ending with dot as number ending with dot
    # replace string begin with dot and space as empty
    splits = [re.sub(rf'^\.\s', '', x).strip() for x in splits]
    # replace string begin with semicolon and space as empty
    splits = [re.sub(rf'^;\s', '', x).strip() for x in splits]
    # discard empty string or space-only
    splits = [x for x in splits if x != '' and x != ' ']

    # split string into list with bullet point characters
    if len(splits) <= 1:
        splits = re.split(bullet_list_symbol, text)
        splits = [x.strip() for x in splits if x != '' and x != ' ']
        if len(splits) > 1:
            return splits
        else:
            return phrase_tokenize(text, nlp)

    # split numbering from next string if the next string starts with numbering pattern and current string is also matches numbering pattern
    remove = []
    iterator = neighborhood(list(splits))
    for prev, curr, nxt in iterator:
        if nxt:
            if re.match(all_pattern, curr) and re.match(startwith_patterns, nxt):
                k = [x for x in re.split(all_pattern, nxt) if x]
                k = [x.strip() for x in k if not re.match(rf'^{roman_number_pattern}$|^{alphabet_pattern}$|^{digit_pattern}$', x, flags=re.IGNORECASE)]
                k = [x for i in k for x in re.split(all_pattern, i) if x]
                k = [x.strip() for x in k if not re.match(rf'^{roman_number_pattern}$|^{alphabet_pattern}$|^{digit_pattern}$', x, flags=re.IGNORECASE)]
                for j in k:
                    splits.insert(splits.index(nxt), j)
                remove.append(nxt)
    for i in remove:
        splits.remove(i)

    tmp_splits = splits.copy()
    iterator = neighborhood(list(tmp_splits))

    # add whitespace into list at current position whenever both current item and next item in the list match numbering pattern (e.g. (i), (a), k) etc.), add whitespace into list at next position (e.g. ['(a)','(b)','XXXX'] -> ['(a)',' ','(b)','XXXX'])
    splits = []
    for prev, curr, nxt in iterator:
        if not prev:
            if not re.match(all_pattern, curr) and re.match(all_pattern, nxt):
                splits.append(' ')
                splits.append(curr)
                continue
        if nxt:
            # if both the current item and next item match list numbering pattern [e.g. (a),(iv) etc.]
            if re.match(all_pattern, curr) and re.match(all_pattern, nxt):
                splits.append(curr)
                splits.append(' ')
            else:
                splits.append(curr)
        else:
            splits.append(curr)

    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    splits = pairwise(splits)

    dic = {}
    primaryLevel = secondaryLevel = tertiaryLevel = quaternaryLevel = quinaryLevel = ''
    last_primary_key = last_secondary_key = last_tertiary_key = last_quaternary_key = last_quinary_key = ''
    last_pattern = ''
    if override_key:
        keys = re.findall(r'(\(*\w+\)*\.*)', override_key)
        if len(keys) <= 4:
            i = 1
            while keys:
                if i == 1:
                    last_primary_key = keys.pop(0)
                    for p in numbering_patterns:
                        if re.match(p, last_primary_key.strip("().")):
                            primaryLevel = p
                            break
                elif i == 2:
                    last_secondary_key = keys.pop(0)
                    for p in numbering_patterns:
                        if re.match(p, last_secondary_key.strip("().")):
                            secondaryLevel = p
                            break
                elif i == 3:
                    last_tertiary_key = keys.pop(0)
                    for p in numbering_patterns:
                        if re.match(p, last_tertiary_key.strip("().")):
                            tertiaryLevel = p
                            break
                elif i == 4:
                    last_quaternary_key = keys.pop(0)
                    for p in numbering_patterns:
                        if re.match(p, last_quaternary_key.strip("().")):
                            quaternaryLevel = p
                            break
                i += 1

    for prev, curr, nxt in neighborhood(list(splits)):
        curr_key, curr_value = curr
        curr_value = tuple(re.split(bullet_list_symbol, curr_value))
        if len(curr_value) == 1:
            curr_value = curr_value[0]
        c_key = curr_key.strip("().")
        if prev is None and override_key is None:
            for p in numbering_patterns:
                if re.match(p, c_key):
                    primaryLevel = p
                    break
            p_key = None
        elif prev is None and override_key:
            p_key = last_primary_key.strip("().")
            for p in numbering_patterns:
                if re.match(p, c_key):
                    secondaryLevel = p
                    break
        else:
            prev_key, prev_value = prev
            p_key = prev_key.strip("().")
        if nxt is not None:
            next_key, next_value = nxt
            n_key = next_key.strip("().")
        else:
            n_key = None

        def isNextNum(prev_num, curr_num):
            # if previous number is None or empty string
            if prev_num is None or prev_num == '':
                return True
            else:
                prev_num = prev_num.strip('()')
            # if two numbers are digits
            if str(curr_num).isdigit() and str(prev_num).isdigit():
                if int(curr_num) == int(prev_num) + 1:
                    return True
            try:
                # if two numbers are alphabet character
                if curr_num == chr(ord(prev_num) + 1):
                    return True
                else:
                    try:
                        prev_num = romanToInt(prev_num)
                        curr_num = romanToInt(curr_num)
                        # if two numbers are roman numbers character
                        if prev_num and curr_num:
                            if curr_num == prev_num + 1:
                                return True
                    except:
                        return False
            except:
                return False

        if re.match(primaryLevel, c_key):
            last_primary_key = curr_key
            last_primary_value = curr_value
        elif re.match(secondaryLevel, c_key):
            last_secondary_key = curr_key
            last_secondary_value = curr_value
        elif re.match(tertiaryLevel, c_key):
            last_tertiary_key = curr_key
            last_tertiary_value = curr_value
        elif re.match(quaternaryLevel, c_key):
            last_quaternary_key = curr_key
            last_quaternary_value = curr_value

        for i, k in enumerate(numbering_patterns):
            if n_key is not None:
                match_c = re.match(k, c_key)
                match_n = re.match(k, n_key)
                if match_c and match_n:
                    match_n = isNextNum(c_key, n_key)
                not_same_pattern = (match_c and (
                    not match_n or match_n is None))
                if not_same_pattern or match_n == False:
                    if k == primaryLevel:
                        for p in numbering_patterns:
                            if p == primaryLevel:
                                continue
                            if re.match(p, n_key):
                                secondaryLevel = p
                                break
                        tmp = {curr_key: {curr_value: {}}}
                    elif k == secondaryLevel:
                        for p in numbering_patterns:
                            if p == secondaryLevel:
                                continue
                            if re.match(p, n_key):
                                tertiaryLevel = p
                                break
                        if re.match(primaryLevel, n_key):
                            tmp = {curr_key: curr_value}
                        else:
                            tmp = {curr_key: {curr_value: {}}}
                    elif k == tertiaryLevel:
                        for p in numbering_patterns:
                            if p == tertiaryLevel:
                                continue
                            if re.match(p, n_key):
                                quaternaryLevel = p
                                break
                        if re.match(primaryLevel, n_key) or re.match(secondaryLevel, n_key):
                            tmp = {curr_key: curr_value}
                        else:
                            tmp = {curr_key: {curr_value: {}}}
                    elif k == quaternaryLevel:
                        for p in numbering_patterns:
                            if p == quaternaryLevel:
                                continue
                            if re.match(p, n_key):
                                quinaryLevel = p
                                break
                        if re.match(primaryLevel, n_key) or re.match(secondaryLevel, n_key) or re.match(tertiaryLevel, n_key):
                            tmp = {curr_key: curr_value}
                        else:
                            tmp = {curr_key: {curr_value: {}}}
                    else:
                        tmp = {curr_key: curr_value}
                    break
            else:
                tmp = {curr_key: curr_value}
                break
            if i == len(numbering_patterns) - 1:
                tmp = {curr_key: curr_value}

        if re.match(primaryLevel, c_key) or (last_pattern == primaryLevel and isNextNum(p_key, c_key)):
            last_pattern = primaryLevel
            if use_nested:
                dic.update(tmp)
            else:
                if curr_key in dic.keys():
                    curr_key += ' '
                dic.update({curr_key: curr_value})
        elif re.match(secondaryLevel, c_key) or (last_pattern == secondaryLevel and isNextNum(p_key, c_key)):
            last_pattern = secondaryLevel
            if use_nested:
                dic[last_primary_key][last_primary_value].update(tmp)
            else:
                if last_primary_key + curr_key in dic.keys():
                    curr_key += ' '
                dic.update({last_primary_key + curr_key: curr_value})
        elif re.match(tertiaryLevel, c_key) or (last_pattern == tertiaryLevel and isNextNum(p_key, c_key)):
            last_pattern = tertiaryLevel
            if use_nested:
                dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value].update(tmp)
            else:
                if last_primary_key + last_secondary_key + curr_key in dic.keys():
                    curr_key += ' '
                dic.update(
                    {last_primary_key + last_secondary_key + curr_key: curr_value})
        elif re.match(quaternaryLevel, c_key) or (last_pattern == quaternaryLevel and isNextNum(p_key, c_key)):
            last_pattern = quaternaryLevel
            if use_nested:
                dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value][last_tertiary_key][last_tertiary_value].update(tmp)
            else:
                if last_primary_key + last_secondary_key + last_tertiary_key + curr_key in dic.keys():
                    curr_key += ' '
                dic.update({last_primary_key + last_secondary_key + last_tertiary_key + curr_key: curr_value})
        elif re.match(quinaryLevel, c_key) or (last_pattern == quinaryLevel and isNextNum(p_key, c_key)):
            last_pattern = quinaryLevel
            if use_nested:
                dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value][last_tertiary_key][
                    last_tertiary_value][last_quaternary_key][last_quaternary_value].update(tmp)
            else:
                if last_primary_key + last_secondary_key + last_tertiary_key + last_quaternary_key + curr_key in dic.keys():
                    curr_key += ' '
                dic.update({last_primary_key + last_secondary_key + last_tertiary_key + last_quaternary_key + curr_key: curr_value})
    return dic


def dic2text(dic, isnested=True):
    import re
    import json
    xyz = r'[xyz]'
    capital_xyz = r'[XYZ]'
    capital_roman_number_pattern = r'(?=[XVI])M*(X[L]|L?X{0,3})(I[XV]|V?I{0,3})'
    roman_number_pattern = r'(?=[xvi])m*(x[l]|l?x{0,3})(i[xv]|v?i{0,3})'
    capital_alphabet_pattern = r'[^\d\sIVXivxa-zXYZ\W_]{1}'
    alphabet_pattern = r'[^\d\sIVXivxA-Zxyz\W_]{1}'
    multi_alphabet_pattern = r'\b[^\d\sIVXivxA-Zxyz\W_]{2}\b'
    # capital_alphabet_pattern = r'[A-Z]{1}'
    # alphabet_pattern = r'[a-z]{1}'
    digit_pattern = r'\b[1-9]\d{0,1}\b'
    multi_dot_digit_pattern = rf'{digit_pattern}\.{digit_pattern}\.{digit_pattern}|{digit_pattern}\.{digit_pattern}'
    patterns = [xyz, capital_xyz, capital_roman_number_pattern, roman_number_pattern,
                capital_alphabet_pattern, alphabet_pattern, multi_alphabet_pattern, multi_dot_digit_pattern, digit_pattern]
    patterns_with_bracket = r'^\(*' + '\)|^\(*'.join(patterns) + r'\)'
    patterns_with_dot = r'^' + '\.\s|'.join(patterns) + r'\.\s'
    all_pattern = '(' + patterns_with_bracket + '|' + patterns_with_dot + ')'
    numbering_with_bracket = r'\(*[^\s\(\)]+\)*\.*'
    text = ''

    def concat_text(text, k, v):
        if not k.strip():
            v += ':'
        elif not v.endswith('.'):
            v += ';'
        if re.match(all_pattern, k):
            k += ' '
        if k.strip():
            text += ' ' + k + v
        else:
            text += v
        return text

    if isnested:
        for k1, v1 in dic.items():
            if isinstance(v1, tuple):
                v1 = '•'.join(v1)
                text = concat_text(text, k1, v1)
            elif isinstance(v1, dict):
                if k1.strip():
                    if re.match(all_pattern, k1):
                        text += ' ' + k1 + ' '
                    else:
                        text += ' ' + k1 + ':'
                for k2, v2 in v1.items():
                    if isinstance(v2, tuple):
                        v2 = '•'.join(v2)
                        text = concat_text(text, k2, v2)
                    elif isinstance(v2, dict):
                        if k2.strip():
                            if re.match(all_pattern, k2):
                                text += ' ' + k2 + ' '
                            else:
                                text += ' ' + k2 + ':'
                        for k3, v3 in v2.items():
                            if isinstance(v3, tuple):
                                v3 = '•'.join(v3)
                                text = concat_text(text, k3, v3)
                            elif isinstance(v3, dict):
                                if k3.strip():
                                    if re.match(all_pattern, k3):
                                        text += ' ' + k3 + ' '
                                    else:
                                        text += ' ' + k3 + ':'
                                for k4, v4 in v3.items():
                                    if isinstance(v4, tuple):
                                        v4 = '•'.join(v4)
                                        text = concat_text(text, k4, v4)
                                    elif isinstance(v4, dict):
                                        if k4.strip():
                                            if re.match(all_pattern, k4):
                                                text += ' ' + k4 + ' '
                                            else:
                                                text += ' ' + k4 + ':'
                                        for k5, v5 in v4.items():
                                            if isinstance(v5, tuple):
                                                v5 = '•'.join(v5)
                                            text = concat_text(text, k5, v5)
                                    else:
                                        text = concat_text(text, k4, v4)
                            else:
                                text = concat_text(text, k3, v3)
                    else:
                        text = concat_text(text, k2, v2)
            else:
                text = concat_text(text, k1, v1)
    else:
        if isinstance(dic, dict):
            for k, v in dic.items():
                k = re.findall(numbering_with_bracket, k)[-1] if k.strip() and re.findall(numbering_with_bracket, k) else k
                if isinstance(v, tuple) or isinstance(v, list):
                    v = '•'.join(v)
                if not v.endswith('.'):
                    text += ' ' + k + ' ' + v + ';'
                else:
                    text += ' ' + k + ' ' + v
        elif isinstance(dic, tuple) or isinstance(dic, list) and all(isinstance(elem, str) for elem in dic):
            text = '•' + '•'.join(dic)
        elif isinstance(dic, tuple) or isinstance(dic, list) and all(isinstance(elem, dict) for elem in dic):
            for elem in dic:
                tmp = '•'
                for k, v in elem.items():
                    k = re.findall(numbering_with_bracket, k)[-1] if k.strip() and re.findall(numbering_with_bracket, k) else k
                    if isinstance(v, tuple) or isinstance(v, list):
                        v = '•'.join(v)
                    if not v.endswith('.'):
                        tmp += ' ' + k + ' ' + v + ';'
                    else:
                        tmp += ' ' + k + ' ' + v
                text += tmp
            
    return text.strip()


def rectify_words_in_string(string):
    import wordninja
    import re
    from textblob import TextBlob

    string = re.sub(r'\s+', '', string)
    string = wordninja.split(string)
    length = len(string)
    string = [TextBlob(word.lower()).correct().raw for word in string]
    string = ' '.join(string).title()

    return string


def replace_weird_multiple_char(my_str):
    import re
    my_str = re.sub(' +', ' ', my_str)
    my_str = re.sub('\({2,}.*', '', my_str)
    my_str = re.sub('\){2,}.*', '', my_str)
    return my_str


def remove_multiple_whitespace(my_str):
    import re
    my_str = re.sub(' +', ' ', my_str)
    return my_str


def remove_punctuation(s):
    '''
    Remove all punctuations in a string
    '''
    import string
    translator = str.maketrans('', '', string.punctuation)
    return s.translate(translator).strip() if s else None


def string_with_whitespace(str_list):
    '''
    insert whitespace that interrupt the string
    :param str_list: list of string
    :return: list of combination of strings interrupted by whitespace
    '''
    def insert_space(string, pos):
        return string[0:pos] + ' ' + string[pos:]
    if isinstance(str_list, str):
        str_list = [str_list]
    result = []
    for idx, s in enumerate(str_list):
        length = len(s)
        for i in range(length):
            result.append(insert_space(s, i).strip())
    return result

def discard_str_with_unwant_char(a_str):
    # To discard string with Chinese characters, Geometric shape and fullwidth ascii variant characters
    import re
    # The 4E00—9FFF range covers CJK Unified Ideographs (CJK=Chinese, Japanese and Korean)
    ChineseRegexp = re.compile(r'[\u4e00-\u9fff]+')
    geometric_shapeRegexp = re.compile(r'[\u25a0-\u25ff]+')
    # fullwidth_ascii_variantsRegexp = re.compile(r'[\uff01-\uff5e]+')

    # or fullwidth_ascii_variantsRegexp.search(a_str) or geometric_shapeRegexp.search(a_str):
    if ChineseRegexp.search(a_str):
        return None
    else:
        return a_str


def filter_invalid_char(my_str):
    # To remove invalid characters from a string
    import re
    my_str = re.sub(' +', ' ', my_str)
    my_str = re.sub('','', my_str)
    my_str = re.sub(r' ([\.,;]) ',r'\1 ',my_str)
    my_str = re.sub(r' ([\)\]])', r'\1', my_str)
    my_str = re.sub(r' ([\(\[]) ',r' \1',my_str)
    my_str = re.sub(r'([\u4e00-\u9fff]+)', '', my_str) # Chinese
    my_str = re.sub(r'([\u2580—\u259f]+)', '_', my_str) # Block Elements
    my_str = re.sub(r'([\u25a0-\u25ff]+)', '_', my_str) # geometric shape
    my_str = re.sub(r'([\ue000—\uf8ff]+)', '_', my_str) # Private Use Area
    my_str = re.sub(r'([\uf06e]+)', '_', my_str)
    my_str = re.sub(r'“',r'"',my_str)
    my_str = re.sub(r'”',r'"',my_str)
    my_str = re.sub(r'’',r"'",my_str)
    # my_str = re.sub(r'([\uff01-\uff5e]+)', '', my_str) # fullwidth ascii variants
    # my_str = re.sub(r'([\u2018\u2019\u201a-\u201d]+)', '', my_str)  # Quotation marks and apostrophe
    return my_str.strip()

def replace_char(my_str):
    import re
    
    my_str = re.sub(' +', ' ', my_str)
    my_str = re.sub('','', my_str)
    my_str = re.sub(r' ([\.,;]) ',r'\1 ',my_str)
    my_str = re.sub(r' ([\)\]])', r'\1', my_str)
    my_str = re.sub(r' ([\(\[]) ',r' \1',my_str)
    my_str = re.sub(r'“',r'"',my_str)
    my_str = re.sub(r'”',r'"',my_str)
    my_str = re.sub(r'’',r"'",my_str)

    return my_str.strip()

def add_space2camel_case(my_str):
    import re
    import string
    
    punct_list = [re.escape(i) for i in string.punctuation]
    neg_lookbehind = rf'(?<!{")(?<!".join(punct_list)})'
    my_str = re.sub(rf"""
        (            # start the group
            # alternative 1
        (?<=[a-z])       # current position is preceded by a lower char
                         # (positive lookbehind: does not consume any char)
        [A-Z]            # an upper char
                         #
        |   # or
            # alternative 2
        (?<!\A)          # current position is not at the beginning of the string
                         # (negative lookbehind: does not consume any char)
        {neg_lookbehind} # ignore in case current position is succeeded by punctuation
        [A-Z]            # an upper char
        (?=[a-z])        # matches if next char is a lower char
                         # lookahead assertion: does not consume any char
        )                # end the group""",
    r' \1', my_str, flags=re.VERBOSE)
    return my_str

def config_nlp_model(nlp):
    '''
    configure special cases into spacy Langugage model in sentence tokenizer
    '''
    from spacy.language import Language
    from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
    from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
    from spacy.util import compile_infix_regex
    from spacy.matcher import PhraseMatcher
    from spacy.language import Language
    from spacy.util import filter_spans
    
    special_abbrev = ["cent."]
    
    @Language.component("set_custom_boundaries")
    def set_custom_boundaries(doc):
        for token in doc[:-1]:
            if token.text in [".(", ").", ".[", "]."]:
                doc[token.i + 1].is_sent_start = True
            elif token.text in [")", "]"]:
                doc[token.i + 1].is_sent_start = False
            elif token.text in ["(", "["]:
                doc[token.i].is_sent_start = False
        return doc

    if "set_custom_boundaries" not in nlp.pipe_names:
        nlp.add_pipe("set_custom_boundaries", before="parser")
        
    @Language.factory("exc_retokenizer")
    class ExceptionRetokenizer:
        def __init__(self, nlp, name="exc_retokenizer"):
            self.name = name
            self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
            for exc in special_abbrev:
                pattern_docs = [
                    nlp.make_doc(text)
                    for text in [exc, exc.upper(), exc.lower(), exc.title()]
                ]
                self.matcher.add("A", pattern_docs)

        def __call__(self, doc):
            with doc.retokenize() as retokenizer:
                for match in filter_spans(self.matcher(doc, as_spans=True)):
                    retokenizer.merge(match)
            return doc
        
    if "exc_retokenizer" not in nlp.pipe_names:
        nlp.add_pipe("exc_retokenizer")

    # Modify tokenizer infix patterns
    infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # ✅ Commented out regex that splits on hyphens between letters:
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
    )

    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer
    
    return nlp

def phrase_tokenize(string, nlp, delimiter_pattern=r'( and/or;|; and/or| and;|; and| or;|; or|;)'):
    '''
    Tokenize string into sentence, then split sentence into phrases with given delimiter pattern
    @param string: string of paragraph
    @type string: str
    @param nlp: spacy nlp Language model object for sentence tokenization
    @type nlp: spacy.load('en_core_web_sm') object
    @param delimiter_pattern: regular expression pattern of DELIMITERS
    @type delimiter_pattern: str
    @rtype string: List[str]
    '''
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    from nltk.tokenize import sent_tokenize
    import re

    string = re.sub(r'No\.|no\.', 'number', string)
    string = re.sub(r'(?<= [.(a-zA-z]{3})\.(?!=(\n))', '', string)
    string = re.sub(r'(?<= [a-zA-z]{2})\.(?!=(\n))', '', string)

    doc = nlp(string)
    string = [sent.text for sent in doc.sents]
    # string = PunktSentenceTokenizer(string).tokenize(string)
    # string = sent_tokenize(string)
    delimiter_pattern2 = delimiter_pattern.replace(' ', '')
    regex = re.compile(delimiter_pattern2)
    string = [[i.strip() for i in re.split(delimiter_pattern, t) if i] for t in string]
    string = [[t2 + ' ' + t[i + 1] if i % 2 == 0 and i < len(t) - 1 else t2 for i, t2 in enumerate(t)] for t in string]
    string = [l for sublist in string for l in sublist if not regex.match(l)]
    if all([i.endswith(':') for i in string]) and len(string)>1: # if there are more than one colon (:), string will split into two phrases, need to combine it back into one string
        string = ' '.join(string)
    if len(string)>1:
        string = [re.sub('number', 'No.', i) for i in string]
    elif len(string) == 1:
        string = string[0]
        string = re.sub('number', 'No.', string)
    return string


def dict_value_phrase_tokenize(dic, nlp, delimiter_pattern=r'( and;|; and| or;|; or|;)'):
    for k, v in dic.items():
        if isinstance(v, str):
            values = phrase_tokenize(
                v, nlp, delimiter_pattern=delimiter_pattern)
        else:
            continue
        if len(values) == 1:
            values = values[0]
        dic[k] = values
    return dic


def flatten(dictionary, parent_key='', separator='_'):
    '''
    flatten a nested dictionary like:
    {'a': {'b': 'XXX', 'c': 'XXXX'}}
    into {'a_b': 'XXX', 'a_c': 'XXXX'}
    where '_' is the default separator and may be customized
    '''
    from collections.abc import MutableMapping
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def get_filepath_with_filename(parent_dir, filename):
    '''
    Given the target filename and a parent directory, traverse file system under the parent directory to lookup for the filename
    return the relative path to the file with filename if match is found
    otherwise raise a FileNotFoundError
    '''
    import errno
    import os

    for dirpath, subdirs, files in os.walk(parent_dir):
        for x in files:
            if x in filename:
                return os.path.join(dirpath, x)
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)


def log_task2csv(csv_name, row):
    '''
    log the record 'row' into path 'csv_name'
    '''
    import csv
    import os.path

    file_exists = os.path.isfile(csv_name)
    # Open the CSV file in "append" mode
    with open(csv_name, 'a', newline='') as f:
        # Create a dictionary writer with the dict keys as column fieldnames
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
            # Append single row to CSV
        writer.writerow(row)

def connect_fill_contour(img):
    # source: https://stackoverflow.com/questions/52004133/how-to-improve-image-quality
    # External libraries used for
    # Image IO
    # from PIL import Image

    # Morphological filtering
    from skimage.morphology import opening
    from skimage.morphology import disk

    # Data handling
    import numpy as np

    # Connected component filtering
    import cv2

    black = 0
    white = 255
    threshold = 254

    # Open input image in grayscale mode and get its pixels.
    # img = Image.open("image.png").convert("LA")
    pixels = np.array(img) # [:,:,0]

    # Remove pixels above threshold
    pixels[pixels > threshold] = white
    pixels[pixels < threshold] = black


    # Morphological opening
    blobSize = 0.6 # Select the maximum radius of the blobs you would like to remove
    structureElement = disk(blobSize)  # you can define different shapes, here we take a disk shape
    # We need to invert the image such that black is background and white foreground to perform the opening
    pixels = np.invert(opening(np.invert(pixels), structureElement))
    
    # Create and save new image.
    # newImg = Image.fromarray(pixels).convert('RGB')
    # newImg.save("newImage.PNG")
    
    return pixels

def remove_black_background(img):
    # source: https://www.geeksforgeeks.org/removing-black-background-and-make-transparent-using-python-opencv/

    import cv2

    # Convert image to image gray 
    if len(img.shape)>2:
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        tmp = img

    # Applying thresholding technique 
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

    # Using cv2.split() to split channels  
    # of coloured image 
    splits = cv2.split(img)

    if len(splits) ==4:
        rgba = splits
    elif len(splits) ==3:
        # Making list of Red, Green, Blue 
        # Channels and alpha 
        rgba = list(splits) + [alpha]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        splits = cv2.split(img)
        rgba = list(splits) + [alpha]

    # Using cv2.merge() to merge rgba 
    # into a coloured/multi-channeled image 
    dst = cv2.merge(rgba, 4)

    return dst

def white_pixels_to_transparent(img):
    # remove white pixels to transparent
    # source: https://stackoverflow.com/questions/55673060/how-to-set-white-pixels-to-transparent-using-opencv

    import cv2
    import numpy as np

    # get the image dimensions (height, width and channels)
    h, w, c = img.shape

    if c == 4: # image is BGRA image
        # Making list of Red, Green, Blue
        # Channels and alpha
        image_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        image_bgra = img
    elif c == 3: # image is BGR image
        image_bgr = img
        # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
        image_bgra = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    else:
        image_bgr = img
        tmp = img
        for _ in range(2):
            # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
            image_bgra = np.concatenate([tmp, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
            tmp = image_bgra
        image_bgra = tmp

    if c < 3:
        # create a mask where white pixels ([255, 255, 255]) are True
        white = np.all(image_bgr == 255, axis=-1)
    else:
        # create a mask where white pixels ([255, 255, 255]) are True
        white = np.all(image_bgr == [255, 255, 255], axis=-1)
    # change the values of Alpha to 0 for all the white pixels
    image_bgra[white, -1] = 0

    return image_bgra
