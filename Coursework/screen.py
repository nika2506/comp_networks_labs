import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Screen:
    def __init__(self, screen_centre, screen_height, spot_diam, move_step):
        self.screen_centre = screen_centre.copy()
        self.screen_height = screen_height
        self.spot_diam = spot_diam
        self.spot_centre = screen_centre.copy()
        self.move_step = move_step

    def getScreenParameters(self):
        return self.screen_centre, self.screen_height

    def findBorders(self):
        left_border = self.screen_centre[0] - self.screen_height / 2
        right_border = self.screen_centre[0] + self.screen_height / 2
        top_border = self.screen_centre[1] + self.screen_height / 2
        bottom_border = self.screen_centre[1] - self.screen_height / 2
        return left_border, right_border, top_border, bottom_border

    def findAvailableAngle(self):
        angle1 = 0
        angle2 = 360
        left_border, right_border, top_border, bottom_border = self.findBorders()

        if self.spot_centre[0] + self.move_step + (self.spot_diam / 2) > right_border:
            angle1 = 90
            angle2 = 270
        if self.spot_centre[0] - self.move_step - (self.spot_diam / 2) < left_border:
            angle1 = 270
            angle2 = 450
        if self.spot_centre[1] + self.move_step + (self.spot_diam / 2) > top_border:
            if angle1 == 0:
                angle1 = 180
                angle2 = 360
            elif angle1 == 90:
                angle1 = 180
            elif angle1 == 270:
                angle2 = 360
        if self.spot_centre[1] - self.move_step - (self.spot_diam / 2) < bottom_border:
            if angle1 == 0:
                angle2 = 180
            elif angle1 == 90:
                angle2 = 180
            elif angle1 == 270:
                angle1 = 360

        return angle1, angle2

    def randMoveSpot(self):
        print('Солнечный зайчик сдвинулся')
        av_angl1, av_angl2 = self.findAvailableAngle()
        angle = random.randint(av_angl1, av_angl2)
        dx = self.move_step * np.cos(angle * np.pi / 180)
        dy = self.move_step * np.sin(angle * np.pi / 180)
        self.spot_centre[0] = self.spot_centre[0] + dx
        self.spot_centre[1] = self.spot_centre[1] + dy

    def makePlotSpot(self):
        left_border, right_border, top_border, bottom_border = self.findBorders()
        figure, axes = plt.subplots()
        axes.vlines(self.screen_centre[0], bottom_border, top_border, color='k')
        axes.hlines(self.screen_centre[1], left_border, right_border, color='k')
        axes.scatter(x=self.spot_centre[0], y=self.spot_centre[1], c='r')
        draw_circle = plt.Circle((self.spot_centre[0], self.spot_centre[1]), self.spot_diam / 2, color='r', alpha=0.5)
        axes.set_aspect(1)
        axes.add_artist(draw_circle)
        plt.xlim(left_border, right_border)
        plt.ylim(bottom_border, top_border)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def processMoveSpot(self, move_iter, show_plots=False):
        #fig = plt.figure()
        for i in range(move_iter):
            if show_plots == True:
                self.makePlotSpot()
            self.randMoveSpot()
        if show_plots == True:
            self.makePlotSpot()
        #ani = FuncAnimation(fig, self.makePlotSpot, interval=100)
        #plt.show()

    def calculateNumOfSquare(self): #находит номер квадрата (сенсора), в котором лежит центр окружности
        #if self.spot_centre[0] == self.screen_centre[0] and self.spot_centre[1] == self.screen_centre[1]:
           # return 0 #значит окружность находится ровно в середине экрана
        if self.spot_centre[0] <= self.screen_centre[0] and self.spot_centre[1] <= self.screen_centre[1]:
            return 1
        if self.spot_centre[0] <= self.screen_centre[0] and self.spot_centre[1] >= self.screen_centre[1]:
            return 2
        if self.spot_centre[0] >= self.screen_centre[0] and self.spot_centre[1] >= self.screen_centre[1]:
            return 3
        if self.spot_centre[0] >= self.screen_centre[0] and self.spot_centre[1] <= self.screen_centre[1]:
            return 4

    def intersectionWithLines(self, centre_square, left_border, right_border, top_border, bot_border):
        rad = self.spot_diam / 2
        intersections_horizontal = []
        perpendic_h = [self.spot_centre[0], self.screen_centre[1]] #точка пересечения перпендикуляра из центра круга с горизонтальной линией
        if (((perpendic_h[1] < self.spot_centre[1] + rad) and (centre_square == 1 or centre_square == 4))
                or ((perpendic_h[1] > self.spot_centre[1] - rad) and (centre_square == 2 or centre_square == 3))):  # если пересечения имеются
            cathetus1 = np.abs(perpendic_h[1] - self.spot_centre[1])
            cathetus2 = np.sqrt(rad ** 2 - cathetus1 ** 2)

            cross1 = [perpendic_h[0] - cathetus2, perpendic_h[1]]
            cross2 = [perpendic_h[0] + cathetus2, perpendic_h[1]]
            intersections_horizontal.append(cross1)
            intersections_horizontal.append(cross2)

        intersections_vertical = []
        perpendic_v = [self.screen_centre[0], self.spot_centre[1]] #точка пересечения перпендикуляра из центра круга с вертикальной линией
        if (((perpendic_v[0] < self.spot_centre[0] + rad) and (centre_square == 1 or centre_square == 2))
                or ((perpendic_v[0] > self.spot_centre[0] - rad) and (centre_square == 3 or centre_square == 4))):
            cathetus1 = np.abs(perpendic_v[0] - self.spot_centre[0])
            cathetus2 = np.sqrt(rad ** 2 - cathetus1 ** 2)
            cross3 = [perpendic_v[0], perpendic_v[1] - cathetus2]
            cross4 = [perpendic_v[0], perpendic_v[1] + cathetus2]
            intersections_vertical.append(cross3)
            intersections_vertical.append(cross4)

        return intersections_horizontal, intersections_vertical

    def intersectionsWithSensors(self):
        print('Расчет пересечений матриц камер с окружностью')
        centre_square = self.calculateNumOfSquare()
        left_border, right_border, top_border, bot_border = self.findBorders()
        intersections_horiz, intersections_vert = self.intersectionWithLines(centre_square,
            left_border, right_border, top_border, bot_border)

        intersections_sensor1 = []
        intersections_sensor2 = []
        intersections_sensor3 = []
        intersections_sensor4 = []

        for point in intersections_horiz:
            if point[0] >= left_border and point[0] <= self.screen_centre[0]:
                intersections_sensor1.append(point)
                intersections_sensor2.append(point)
            if point[0] >= self.screen_centre[0] and point[0] <= right_border:
                intersections_sensor3.append(point)
                intersections_sensor4.append(point)
        for point in intersections_vert:
            if point[1] >= bot_border and point[1] <= self.screen_centre[1]:
                intersections_sensor1.append(point)
                intersections_sensor4.append(point)
            if point[1] >= self.screen_centre[1] and point[1] <= top_border:
                intersections_sensor2.append(point)
                intersections_sensor3.append(point)

        return (intersections_sensor1, intersections_sensor2, intersections_sensor3,
                intersections_sensor4, centre_square)

    def makePlotSpotAndIntersections(self):
        centre_square = self.calculateNumOfSquare()
        left_border, right_border, top_border, bot_border = self.findBorders()
        horiz, vert = self.intersectionWithLines(centre_square, left_border,
                                                 right_border, top_border, bot_border)
        horiz_x = []
        horiz_y = []
        for i in range (len(horiz)):
            horiz_x.append(horiz[i][0])
            horiz_y.append(horiz[i][1])
        vert_x = []
        vert_y = []
        for i in range (len(vert)):
            vert_x.append(vert[i][0])
            vert_y.append(vert[i][1])

        figure, axes = plt.subplots()
        axes.vlines(self.screen_centre[0], bot_border, top_border, color = 'k')
        axes.hlines(self.screen_centre[1], left_border, right_border, color = 'k')
        axes.scatter(x=self.spot_centre[0], y=self.spot_centre[1], c='r')
        draw_circle = plt.Circle((self.spot_centre[0], self.spot_centre[1]), self.spot_diam / 2, color='r', alpha=0.5)
        axes.set_aspect(1)
        axes.add_artist(draw_circle)
        axes.scatter(x=horiz_x, y=horiz_y, c='b')
        axes.scatter(x=vert_x, y=vert_y, c='g')
        plt.xlim(left_border, right_border)
        plt.ylim(bot_border, top_border)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()