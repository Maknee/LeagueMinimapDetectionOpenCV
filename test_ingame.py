import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import PIL

from screen_capture import capture_screenshot
from detection import DetectionManager

icons_folder = 'icons'
minimap_ratio = 800 / 1080  # How big is the minimap width?
icon_ratio = 25 / 280  # 280 is the width/height of icons
icon_search_ratio = 0.5  # percentage of the champion icon to be used in matchTEmplate
threshold = 0.6  # threshold for matchTemplate


def main():
    dm = DetectionManager(icons_folder, minimap_ratio, icon_ratio, icon_search_ratio, threshold)

    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2)
    ax3 = fig.add_subplot(2, 4, 3)
    ax4 = fig.add_subplot(2, 4, 4)
    ax5 = fig.add_subplot(2, 4, 5)
    ax6 = fig.add_subplot(2, 4, 6)
    ax7 = fig.add_subplot(2, 4, 7)
    ax8 = fig.add_subplot(2, 4, 8)

    ax1.set_title('minimap')
    ax2.set_title('red filter')
    ax3.set_title('red contours')
    ax4.set_title('red detection')
    ax5.set_title('minimap')
    ax6.set_title('blue filter')
    ax7.set_title('blue contours')
    ax8.set_title('blue detection')

    screenshot = capture_screenshot()
    minimap, icon_size = dm.get_minimap_and_icon_size(screenshot)
    icons = dm.get_icons(icon_size)

    while True:
        screenshot = capture_screenshot()
        minimap, icon_size = dm.get_minimap_and_icon_size(screenshot)

        red_filter_minimap = dm.filter_red(minimap)
        blue_filter_minimap = dm.filter_blue(minimap)
        red_contours_minimap, red_final_minimap = dm.find_champions(minimap, icons, red_filter_minimap, [0, 255, 0])
        blue_contours_minimap, blue_final_minimap = dm.find_champions(minimap, icons, blue_filter_minimap, [0, 255, 0])

        ax1.imshow(minimap, interpolation='bilinear')
        ax2.imshow(red_filter_minimap, interpolation='bilinear')
        ax3.imshow(red_contours_minimap, interpolation='bilinear')
        ax4.imshow(red_final_minimap, interpolation='bilinear')
        ax5.imshow(minimap, interpolation='bilinear')
        ax6.imshow(blue_filter_minimap, interpolation='bilinear')
        ax7.imshow(blue_contours_minimap, interpolation='bilinear')
        ax8.imshow(blue_final_minimap, interpolation='bilinear')

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.tight_layout()
        plt.pause(0.0001)


if __name__ == "__main__":
    main()
