import numpy as np
from matplotlib.pyplot import *
from scipy.spatial import Delaunay
import cv2

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges
def find_edges_with(i, edge_set):
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second

def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i,j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst


def croppedmatchingareas(clientimagepath, goodclustersarray):
    #image = cv2.imread(clientimagepath)
    image = clientimagepath

    # create a mask with white pixels
    mask = np.ones(image.shape, dtype=np.uint8)
    mask.fill(255)
    Newclusters = []
    for i in goodclustersarray:
        for j in i:
            Newclusters.append(j)

    arrays = np.array(Newclusters)
    # points to be cropped
    roi_corners = np.array([arrays], dtype=np.int32)
    # fill the ROI into the mask
    cv2.fillPoly(mask, roi_corners, 0)

    # The mask image
    cv2.imwrite('image_masked.jpg', mask)

    # applying th mask to original image
    masked_image = cv2.bitwise_or(image, mask)

    # The resultant image
    cv2.imwrite('new_masked_image.jpg', masked_image)
    return masked_image


from functions import DB_SCAN

def function():
    img=cv2.imread("115.jpg")
    # Constructing the input point data
    images = np.array(img)
    img1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1, None)
    clusters = DB_SCAN(kp1,20)
    x=[]
    y=[]

    for i in clusters:
        for j in i:
            x=np.append(x,j[0])
            y = np.append(y, j[1])


    inside = ((x ** 2 + y ** 2 > 1.0) & ((x - 3) ** 2 + y ** 2 > 1.0))
    points = np.vstack([x[inside], y[inside]]).T
    #print(points)
    # Computing the alpha shape
    edges = alpha_shape(points, alpha=20, only_outer=True)

    ed=stitch_boundaries(edges)

    '''
    from matplotlib.pyplot import *
    
    # Constructing the input point data
    np.random.seed(0)
    x = 3.0 * np.random.rand(2000)
    y = 2.0 * np.random.rand(2000) - 1.0
    inside = ((x ** 2 + y ** 2 > 1.0) & ((x - 3) ** 2 + y ** 2 > 1.0))
    points = np.vstack([x[inside], y[inside]]).T
    
    # Computing the alpha shape
    edges = alpha_shape(points, alpha=0.25, only_outer=True)
    '''
    # Plotting the output
    figure()
    axis('equal')
    plot(points[:, 0], points[:, 1], '.')
    for i, j in edges:
        plot(points[[i, j], 0], points[[i, j], 1])
    show()


    # original image
    # -1 loads as-is so if it will be 3 or 4 channel as the original
    image = cv2.imread('cropped.jpg', -1)
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    print(ed[0])

    roi_corners = np.array([ed[0]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # save the result
    cv2.imwrite('image_masked2.jpg', masked_image)

def function2(clusters1):
    #img = cv2.imread("100.jpg")
    # Constructing the input point data
    #images = np.array(img)
    #img1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    #surf = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    #kp1, des1 = surf.detectAndCompute(img1, None)
    clusters = clusters1
    x = []
    y = []

    for i in clusters:
        for j in i:
            x = np.append(x, j[0])
            y = np.append(y, j[1])

    inside = ((x ** 2 + y ** 2 > 1.0) & ((x - 3) ** 2 + y ** 2 > 1.0))
    points = np.vstack([x[inside], y[inside]]).T
    # print(points)
    # Computing the alpha shape
    edges = alpha_shape(points, alpha=20, only_outer=True)
    ed = stitch_boundaries(edges)
    '''
    from matplotlib.pyplot import *

    # Constructing the input point data
    np.random.seed(0)
    x = 3.0 * np.random.rand(2000)
    y = 2.0 * np.random.rand(2000) - 1.0
    inside = ((x ** 2 + y ** 2 > 1.0) & ((x - 3) ** 2 + y ** 2 > 1.0))
    points = np.vstack([x[inside], y[inside]]).T

    # Computing the alpha shape
    edges = alpha_shape(points, alpha=0.25, only_outer=True)
    '''
    # Plotting the output
    figure()
    axis('equal')
    plot(points[:, 0], points[:, 1], '.')
    for i, j in edges:
        plot(points[[i, j], 0], points[[i, j], 1])
    show()

    # original image
    # -1 loads as-is so if it will be 3 or 4 channel as the original
    image = cv2.imread('115.jpg', -1)
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    print(ed[0])

    roi_corners = np.array([ed[0]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # save the result
    cv2.imwrite('image_masked2.jpg', masked_image)
    return masked_image