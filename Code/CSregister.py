# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:30:40 2021

@author: 4105664
"""

from enum import Enum
from functools import cached_property
from itertools import product
from pathlib import Path

import attr
import cv2
import os
import numpy as np
import typer
from loguru import logger

class Detection:
    def __init__(self, ObjectsPath, Objects, DetectionModels, ThresholdPercentage=20):
        self.ObjectsPath = ObjectsPath
        self.Objects = Objects
        self.DetectionModels = DetectionModels
        self.ThresholdPercentage = ThresholdPercentage

@attr.s
class Image:
    """
    Either a raw image or target image used in the registration process
    """

    path = attr.ib()

    def create_keypoints_descriptors(self, max_features=1000):
        """
        Create keypoints and descriptors used for feature matching
        """

# TODO check if mask is performed after creating features

        logger.debug(f"Generating keypoints for {self.filename}")
        array_gray = cv2.cvtColor(self.array, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(max_features)
        keypoints, descriptors = orb.detectAndCompute(array_gray, mask = self.mask)

        self.keypoints = keypoints
        self.descriptors = descriptors

    def add_mask_from_file(self, mask_file):
        """
        Mask the area for which we want to generate keypoints using a file. File
        should be a png, where features will not be generated at locations where
        pixels are transparent.
        """

        # Load image (IMREAD_UNCHANGED needed to read the transparency layer)
        self.mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

        # Resize mask to match image dimensions
        dims = (self.array.shape[1], self.array.shape[0])
        self.mask = cv2.resize(self.mask, dims, interpolation=cv2.INTER_AREA)

        # Mask locations where the mask is not transparent
        self.mask = np.array(self.mask[:, :] == 255, dtype=np.uint8)
        
        self.mask = self.mask[:,:]
    
    def add_mask(self, mask):
        """
        Mask the area for which we want to generate keypoints using a file. File
        should be a png, where features will not be generated at locations where
        pixels are transparent.
        """
        self.mask = mask
        

    def add_mask_by_area(self, area):
        """
        Mask an area of the image from generating keypoints. Typically used because
        we only want to use the top half on an image to generate keypoints since
        these are more likely to be stable (i.e. on the horizon).
        """

        # Create blank mask. Zero values are ignored from keypoint feature creation
        mask = np.zeros((self.height, self.width))
        mask = mask.astype(np.uint8)

        if area:

            areas = [x.strip() for x in area.split(",")]

            for a in areas:

                if a == "upper third":
                    mask[: int(mask.shape[0] / 3), :] = 1
                elif a == "lower third":
                    mask[-int(mask.shape[0] / 3) :, :] = 1
                elif a == "upper half":
                    mask[: int(mask.shape[0] / 2), :] = 1
                elif a == "lower half":
                    mask[-int(mask.shape[0] / 2) :, :] = 1
                elif a == "left half":
                    mask[:, : int(mask.shape[1] / 2)] = 1
                elif a == "right half":
                    mask[:, -int(mask.shape[1] / 2) :] = 1
                elif a == "left third":
                    mask[:, : int(mask.shape[1] / 3)] = 1
                elif a == "right third":
                    mask[:, -int(mask.shape[1] / 3) :] = 1
                elif a == "lower left quarter":
                    mask[-int(mask.shape[0] / 2) :, : int(mask.shape[1] / 2)] = 1
                elif a == "upper right quarter":
                    mask[: int(mask.shape[0] / 2), -int(1 * mask.shape[1] / 2) :] = 1
                elif a == "lower right quarter":
                    mask[-int(mask.shape[0] / 2) :, -int(mask.shape[1] / 2) :] = 1
                elif a == "upper left quarter":
                    mask[: int(mask.shape[0] / 2), : int(1 * mask.shape[1] / 2)] = 1
                else:
                    logger.warning(f"Unknown area mask option: {a}")
        else:
            # If no areas given, use entire image
            mask[:, :] = 1

        # Mask nominal 10 pixels from each edge
        n_edge = 10
        mask[:, :n_edge] = 0
        mask[:, -n_edge:] = 0
        mask[:n_edge, :] = 0
        mask[-n_edge:, :] = 0

        self.mask = mask

    def load_array(self):
        """
        Loads the data from the image file and store in a numpy array.
        """

        array = cv2.imread(str(self.path))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2BGRA)
        self.array = array

    def resize(self, height, width):
        """
        Resize the image data to a specific width and height. Typically used to
        ensure all images have same dimensions.
        """
        self.array = cv2.resize(
            self.array, (width, height), interpolation=cv2.INTER_AREA
        )

    @property
    def filename(self):
        return Path(self.path).name

    @property
    def height(self):
        return self.array.shape[0]

    @property
    def width(self):
        return self.array.shape[1]
    
@attr.s
class ImageComparer:
    """
    Used to compare and match keypoints between a raw image and target image
    """

    raw_image = attr.ib()
    target_image = attr.ib()
    warp = attr.ib()

    def __attrs_post_init__(self):
        # Try performing the warp on initilization
        # https://www.attrs.org/en/stable/init.html#post-init-hook
        try:
            self.apply_warp()
        except:
            pass

    @cached_property
    def matches(self, max_distance=150):
        """
        Find keypoint matches between the raw image and target image
        """

        logger.debug(
            f"Finding matches between {self.raw_image.filename} and {self.target_image.filename}"
        )

        # Brute force matching works best.
        # Tried with FLANN and with KNN, but did not give as good results.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        all_matches = bf.match(
            self.raw_image.descriptors, self.target_image.descriptors
        )
        matches = []
        for m in all_matches:
            if m.distance < max_distance:
                matches.append(m)

        # Matches should be in the same region of the image. Use the threshold to set
        # how far matches can be away from each other (spatially) before discarding
        # as a bad match
        max_dim = max(self.target_image.height, self.target_image.width)
        threshold = 0.2

        # See how far keypoints are spatially. Iterate backwards since we are
        # removing items from the list
        for m in matches[::-1]:

            raw_x, raw_y = self.raw_image.keypoints[m.queryIdx].pt
            target_x, target_y = self.target_image.keypoints[m.trainIdx].pt

            spatial_distance = np.sqrt(
                (raw_x - target_x) ** 2 + (raw_y - target_y) ** 2
            )

            if spatial_distance > max_dim * threshold:
                matches.remove(m)

        return matches

    @cached_property
    def score(self):
        """
        Calculate score based on h_det, the determinatate of the transformation
        matrix. Best score is 0, worst score is positive infinity. Scores cannot be
        less than zero

        """

        if not hasattr(self, "h_det"):
            logger.warning(
                f"Could not find transformation between "
                f"{self.raw_image.filename} and {self.target_image.filename}"
            )
            return 999

        return abs(1 - self.h_det)

    def apply_warp(self):
        """
        Registers the raw image onto the target image
        """

        # Get key points from raw image and target image
        raw_pts = np.float32(
            [self.raw_image.keypoints[m.queryIdx].pt for m in self.matches]
        )
        target_pts = np.float32(
            [self.target_image.keypoints[m.trainIdx].pt for m in self.matches]
        )

        # Try estimate warp parameters depending on method specified
        if self.warp == "affine":
            h, mask = cv2.estimateAffinePartial2D(raw_pts, target_pts)
            self.registered_array = cv2.warpAffine(
                self.raw_image.array,
                h,
                (self.target_image.width, self.target_image.height),
            )
            self.h_det = np.linalg.det(h[:2, :2])

        elif self.warp == "perspective":
            h, mask = cv2.findHomography(
                raw_pts, target_pts, cv2.RANSAC, ransacReprojThreshold=10
            )
            self.registered_array = cv2.warpPerspective(
                self.raw_image.array,
                h,
                (self.target_image.width, self.target_image.height),
            )
            self.h_det = np.linalg.det(h)

        # Store results
        self.h = h
        self.matches_mask = mask.ravel().tolist()

    def save_registered_image(self, folder):
        """
        Saves the registered image to the specified folder using the same filename as
        the raw image.
        """
        filename = f"{self.raw_image.path.stem}-registered.jpg"
        filepath = Path(folder, filename)
        cv2.imwrite(str(filepath), self.registered_array)

        logger.info(f"Saved registered image: {filename}")
        self.registered_image_path = filepath

    def save_debug_image(self, folder, draw_outliers=False):
        """
        Saves an image showing the keypoint matches to the specified folder.
        """

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0))

        if draw_outliers:
            draw_params["matchesMask"] = self.matches_mask

        # Draw debug plot
        img_debug = cv2.drawMatches(
            self.raw_image.array,
            self.raw_image.keypoints,
            self.target_image.array,
            self.target_image.keypoints,
            self.matches,
            None,
            **draw_params,
        )

        # Resize debug plot
        resize_factor = 3
        debug_height, debug_width = [
            int(x / resize_factor) for x in img_debug.shape[:2]
        ]
        img_debug = cv2.resize(
            img_debug, (debug_width, debug_height), interpolation=cv2.INTER_AREA
        )

        filename = f"{self.raw_image.path.stem}-debug.jpg"
        cv2.imwrite(str(Path(folder, filename)), img_debug)

    @property
    def stats_str(self):
        """
        Generates a summary string of the match.
        """

        return ",".join(
            [
                f"{self.registered_image_path}",
                f"{self.score:.2f}",
                f"{self.h_det:.4f}",
                f"{len(self.matches)}",
            ]
        )

class RegistrationStatsCsv:
    """
    A file which stores a summary of the registration statistics. Can be used to
    check the homography determinate to check how large of a transformation was
    performed on the images.
    """

    filename = attr.ib()

    def reset(self):
        with open(self.filename, "w") as f:
            f.write("registered_image_path,score,h_det,n_matches\n")

    def write_line(self, s):
        with open(self.filename, "a") as f:
            f.write(f"{s}\n")

class Warp(str, Enum):
    """
    Provide options for how the algorithim can do the registration.

    From testing, the perspective option is better when there are a variety of
    cameras, i.e. when lots of different people take images in a CoastSnap cradle.
    Perspective has more degrees of freedom it can solve for, but that can result in
    sometimes unstable transformations (i.e. the algorithm would fit a large skew
    when it is obviously unrealistic). Ideally it would be good if we could limit
    some of the degrees of freedom with the perspective option, but I have not been
    able to do this so far.

    The affine option is better when it is a single camera in a relatively fixed
    location (i.e. trail cam taking photos every 15 mins). This algorithm has less
    degrees of freedom so offers a more stable transformation, but cannot do the
    large corrections needed if different lens or camera positions are used.

    Refs:
        https://stackoverflow.com/a/45644845
        https://typer.tiangolo.com/tutorial/parameter-types/enum/
    """

    perspective = "perspective"
    affine = "affine"
            
def register_img(
    image: str,
    imagePath= os.getcwd(),
    mask = None,
    maskPath = os.getcwd(),
    targetPath = os.getcwd(),
    warp: Warp = typer.Option(
        Warp.perspective, case_sensitive=False, help="type of transformation to apply"
    ),
    ): 
    
    """
    Registers raw image to target images.
    """

    # TODO Check if target folders exist
    # TODO Check if folders exist
    # TODO Check for registered images in filtered folder

    targetMask = os.path.join(targetPath,'target_mask.png')

    # Get all target images
    target_img_paths = [x for x in Path(targetPath).glob("target*.jpg")]
    targets = []
    for target_img_path in target_img_paths:
        target = Image(target_img_path)
        target.load_array()
        target.add_mask_from_file(targetMask)
        target.create_keypoints_descriptors()
        targets.append(target)
        

    # Read raw image
    raw = Path(imagePath,image)
    rawout = []
    r = Image(raw)
    r.load_array()
    r.resize(height=target.height, width=target.width)
    r.add_mask(mask)
    r.create_keypoints_descriptors()
    rawout.append(r)

    # Store all image matches with target images
    image_matches = [ImageComparer(r, t, warp) for r, t in product(rawout, targets)]
    
    # Find best match and apply warp
    best_match = min(image_matches, key=lambda x: x.score)
    
    # Return best match registration
    return cv2.cvtColor(best_match.registered_array, cv2.COLOR_BGRA2RGB)

            