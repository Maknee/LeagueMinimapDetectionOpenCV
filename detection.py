import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


class DetectionManager:
    def __init__(self, icons_folder, minimap_ratio, icon_ratio, icon_search_ratio, threshold):
        """Initializes fields for dectection manager

        Args:
            icons_folder (string): path to icons folder
            minimap_ratio (float): ratio of minimap to screen
            icon_ratio (float): ratui if icon to minimap
            icon_search_ratio (float): how much of the icon should we search
            threshold (float): threshold of how much matchtemplate should be matched with
        """
        self.icons_folder = icons_folder
        self.minimap_ratio = minimap_ratio
        self.icon_ratio = icon_ratio
        self.icon_search_ratio = icon_search_ratio
        self.threshold = threshold

    def get_minimap_and_icon_size(self, screenshot):
        """returns a tuple containing the minimap in the screenshot and the icon size based on the minimap width

        Args:
            screenshot (np.array): numpy array containing the screenshot

        Returns:
            (np.array, int): minimap as numpy array, icon size
        """
        minimap_shape = tuple(int(v * self.minimap_ratio) for v in screenshot.shape)
        minimap_x, minimap_y = minimap_shape[0], minimap_shape
        minimap_size = screenshot.shape[0] - minimap_x
        minimap = screenshot[-minimap_size:, -minimap_size:]

        icon_size = int(minimap_size * self.icon_ratio)

        return (minimap, icon_size)

    def get_icons(self, icon_size):
        """Parses icons inside the icons folder, resizes/crops images and returns an array of the images as numpy arrays

        Args:
            icon_size (int): icon_size returned by get_minimap_and_icon_size

        Returns:
            [np.array]: array of np.array representing icons
        """
        icons = []
        for p in os.listdir(self.icons_folder):
            champion = p[:p.find('.')]
            extension = p[p.rfind('.'):]
            if extension != '.png':
                continue

            p = os.path.join(self.icons_folder, p)
            if not os.path.isfile(p):
                continue
            img = cv2.imread(p, cv2.IMREAD_COLOR)

            w, h, _ = img.shape

            # switch to rgb
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])

            # resize to correct icon size
            img = cv2.resize(img, dsize=(icon_size, icon_size), interpolation=cv2.INTER_CUBIC)

            # crop
            w, h = img.shape[0], img.shape[1]
            l = int(w * self.icon_search_ratio / 2)
            r = w - l
            t = int(h * self.icon_search_ratio / 2)
            b = h - t
            img = img[l:r, t:b]

            icons.append((champion, img))

        return icons

    def filter_red(self, minimap):
        """filters only red colors in minimap

        Args:
            minimap (np.array): np.array of the minimap

        Returns:
            np.array: minimap with red filter
        """
        lower_red = np.array([100, 20, 20])
        upper_red = np.array([255, 100, 100])
        img = cv2.inRange(minimap, lower_red, upper_red)
        kernel = np.ones((5, 5), np.uint8)
        #img = cv2.erode(img, kernel, iterations = 1)
        #img = cv2.dilate(img,kernel,iterations = 1)
        #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        #img = cv2.dilate(img, kernel, iterations = 1)
        #img = cv2.erode(img, kernel, iterations = 1)
        return img

    def filter_blue(self, minimap):
        """filters only blue colors in minimap

        Args:
            minimap (np.array): np.array of the minimap

        Returns:
            np.array: minimap with blue filter
        """
        lower_red = np.array([0, 20, 100])
        upper_red = np.array([30, 170, 255])
        img = cv2.inRange(minimap, lower_red, upper_red)
        return img

    def find_champions(self, minimap, icons, color_filtered_minimap, box_color=[0, 255, 255]):
        """Finds all the champions using contours to find selected regions in color filtered minimap

        Args:
            minimap (np.array): np.array of the minimap
            icons ([np.array]): array of np.arrays of icons
            color_filtered_minimap (np.array): filtered color of the minimap
            box_color (list, optional): box color when the champion is detected. Defaults to [0, 255, 255].

        Returns:
            (np.array, np.array): (contoured image, minimap with detected champions on top)
        """
        height, width, _ = minimap.shape
        contours_minimap = np.copy(minimap)
        detected_champions_minimap = np.copy(minimap)

        contours, hierarchy = cv2.findContours(color_filtered_minimap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            (x, y), r = cv2.minEnclosingCircle(c)  # find circle enclosing the contour
            center = (int(x), int(y))
            radius = int(r)
            x, y = int(x) - radius, int(y) - radius
            w = radius * 2
            h = radius * 2

            # some checks to make sure the circle size is big enough to be a champion icon
            if r > 12 and r < 40 and x >= 0 and x + w < width and y >= 0 and y + h < height:
                cv2.circle(contours_minimap, center, radius, (0, 255, 0), 2)

                c_x = max(x - 0, 0)
                c_w = min(x + w + 0, width)
                c_y = max(y - 0, 0)
                c_h = min(y + h + 0, height)
                c = minimap[c_y:c_h, c_x:c_w]

                # find the best champion that has the highest match
                top_champion = ''
                val = 0
                for champion, icon in icons:
                    res = cv2.matchTemplate(c, icon, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= self.threshold)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    for x1, y1 in zip(*loc[::-1]):
                        if max_val > val:
                            val = max_val
                            top_champion = champion
                            top_draw = (x + x1, y + y1, icon.shape[0], icon.shape[1])

                if top_champion != '':
                    # print(top_champion)
                    x, y, w, h = top_draw
                    cv2.putText(detected_champions_minimap, top_champion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.rectangle(detected_champions_minimap, (x, y), (x+w, y+h), box_color, 1)

        return contours_minimap, detected_champions_minimap
