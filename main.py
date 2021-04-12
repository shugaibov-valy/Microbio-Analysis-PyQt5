#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QByteArray, Qt, QRectF, QLineF, pyqtSignal, QTimer
from PyQt5.QtGui import (QFontDatabase, QFont, QPainter, 
                         QPainterPath, QColor, QPen)
from PyQt5.QtWidgets import (QPushButton, QApplication, QWidget,
                             QDialog, QVBoxLayout)


# In[2]:


import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import colorsys


from cv_color_hsv import count_circles, target_hsv


FONT = b'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'


class Ui_MainWindow(object):
    
    #  Функция для первоначальной настройки параметров окна    
    def setupUi(self, MainWindow):
        # Параметры окна
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1095, 799)
        MainWindow.setStyleSheet("background-color: rgb(47, 34, 34);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.setWindowFlag(QtCore.Qt.MSWindowsFixedSizeDialogHint, True)
        
        font = QtGui.QFont()
        font.setPointSize(10)
        
        #  Параметры для виджета отображения фото
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 601, 601))
        self.pixmap = QPixmap('app_images/no_image.jpg')      # no name image
        self.label.setPixmap(self.pixmap)
        self.label.setObjectName("label")
        
       #
    
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(660, 270, 291, 131))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        #
        
        #  Параметры для кнопки загрузки теста
        
        self.button_load_photo = QtWidgets.QPushButton(self.centralwidget)
        self.button_load_photo.setGeometry(QtCore.QRect(60, 670, 500, 31))
        self.button_load_photo.setFont(font)
        self.button_load_photo.setStyleSheet("background-color: rgb(0, 0, 0); color: white;")
        self.button_load_photo.setObjectName("pushButton")
        
        #  Параметры для надписи выберите цвет
        
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(660, 400, 351, 39))
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(660, 230, 311, 41))
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(810, 45, 311, 41))
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        
        #  Кнопка вывода информации
        
        self.button_result = QtWidgets.QPushButton(self.centralwidget)
        self.button_result.setGeometry(QtCore.QRect(660, 460, 381, 28))
        self.button_result.setStyleSheet("background-color: rgb(0, 0, 0); color: white;")
        self.button_result.setFont(font)
        self.button_result.setObjectName("button_result")
        
        #  Кнопка вывода информации
        #  H
        self.spinBox = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.spinBox.setFont(font)
        self.spinBox.setObjectName("spinBox")
        self.spinBox.setStyleSheet("background-color: rgb(0, 0, 0); color: white;")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinBox)
        self.spinBox.setRange(0, 360)
        #   S       
        self.spinBox_2 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.spinBox_2.setFont(font)
        self.spinBox_2.setObjectName("spinBox_2")
        self.spinBox_2.setStyleSheet("background-color: rgb(0, 0, 0); color: white;")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinBox_2)
        self.spinBox_2.setRange(0, 255)
        #  V
        self.spinBox_3 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.spinBox_3.setFont(font)
        self.spinBox_3.setObjectName("spinBox_3")
        self.spinBox_3.setStyleSheet("background-color: rgb(0, 0, 0); color: white;")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.spinBox_3)
        self.spinBox_3.setRange(0, 255)
        
        #  ЧЕК_БОКС для выбора цвета по умолчанию
        
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(660, 180, 185, 51))
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        
        #  Кнопка сохранения
        
        self.button_save = QtWidgets.QPushButton(self.centralwidget)
        self.button_save.setGeometry(QtCore.QRect(60, 730, 500, 31))
        self.button_save.setFont(font)
        self.button_save.setObjectName("button_save")
        self.button_save.setStyleSheet("background-color: rgb(0, 0, 0); color: white;")
        
        ##### ПИПЕТКА #####
        self.view = CColorStraw(self.centralwidget)
        self.view.setGeometry(QtCore.QRect(950, 25, 100, 100))
        self.view.setFont(font)
        self.view.setStyleSheet("background-color: white;")
        self.view.setObjectName("pipetka") 
        self.view.setIcon(QIcon('app_images/pipetka.png'))
        self.view.setIconSize(QSize(75, 75))
        ###################
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1095, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Подсчёт колоний микроорганизмов"))
        
        #  Кнопка загрузки теста
        
        self.button_load_photo.setText(_translate("MainWindow", "Загрузить тест"))
        self.button_load_photo.clicked.connect(self.open_img)
        
        #  Кнопка вывода информации
        
        self.button_result.setText(_translate("MainWindow", "Найти колонии"))
        self.button_result.clicked.connect(self.generation_img)
        
        #  Чек_бокс для автоматического выбора цвета
        
        self.checkBox.setText(_translate("MainWindow", "Цвет по умолчанию"))
        self.checkBox.stateChanged.connect(self.changeTarget)
        
        #  Кнопка сохранения теста
        
        self.button_save.setText(_translate("MainWindow", "Сохранить тест"))
        self.button_save.clicked.connect(self.save_img)
        
        #  
        
        
        #
        self.label_7.setText(_translate("MainWindow", "Выбрать цвет:"))
        self.label_6.setText(_translate("MainWindow", "Настройка цвета вручную:"))
        self.label_5.setText(_translate("MainWindow", "Обнаружено колоний:"))
        self.label_2.setText(_translate("MainWindow", "H:"))
        self.label_3.setText(_translate("MainWindow", "S:"))
        self.label_4.setText(_translate("MainWindow", "V:"))
        #
        
        self.pixmap_img = ''              # file_name of pixmap img
    
    
    def open_img(self):       # загрузка изображения в окно
        self.file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Images (*png)')
        if self.file_name:
            self.pixmap = QPixmap(self.file_name)      # вставка изображения
            self.label.setPixmap(self.pixmap)
            self.pixmap_img = self.file_name
            
            # Clear widgets' data
            self.checkBox.setChecked(False)
            self.spinBox.setDisabled(False)
            self.spinBox_2.setDisabled(False)
            self.spinBox_3.setDisabled(False)
            self.label_5.setText('Обнаружено колоний:')
            
    def changeT(self, color=[0, 0, 0]):
        self.spinBox.setValue(color[0])
        self.spinBox_2.setValue(color[1])
        self.spinBox_3.setValue(color[2])
    
    
    def changeTarget(self, state, color=(0, 0, 0)):          # слежка за CheckBox
        if state == Qt.Checked:
            self.label_5.setText('Обнаружено колоний:')
            self.spinBox.setDisabled(True)
            self.spinBox_2.setDisabled(True)
            self.spinBox_3.setDisabled(True)
            
            try:
                #From array to list
                self.arr_hsv_numpy = target_hsv(self.file_name)  
                self.arr_hsv_list = self.arr_hsv_numpy.tolist()
            
                # Set HSV in SpinBox
                self.spinBox.setValue(self.arr_hsv_list[0] * 2)
                self.spinBox_2.setValue(self.arr_hsv_list[1])
                self.spinBox_3.setValue(self.arr_hsv_list[2])
            except:
                pass
            
        else:
            self.spinBox.setDisabled(False)
            self.spinBox_2.setDisabled(False)
            self.spinBox_3.setDisabled(False)
            
            # Set zero in SpinBox
            self.spinBox.setValue(color[0])
            self.spinBox_2.setValue(color[1])
            self.spinBox_3.setValue(color[2])
                    
    def generation_img(self):      # кнопка подсчета кружков и контуры
        try:
            if self.checkBox.isChecked() == True:
                self.label_5.setText(f'Обнаружено колоний: {count_circles(self.file_name, self.arr_hsv_list)}')   # target color - True
            else:
                hsv = [int(self.spinBox.text()) / 2, int(self.spinBox_2.text()), int(self.spinBox_3.text())]    # list hsv of spinBox
                self.label_5.setText(f'Обнаружено колоний: {count_circles(self.file_name, hsv)}')   # target color - False
            self.pixmap = QPixmap('original.png')      # вставка изображения
            self.label.setPixmap(self.pixmap)
            self.pixmap_img = 'original.png'
        except:
            pass
        
    def save_img(self):             # кнопка сохранения
        try:
            way_to_save = QFileDialog.getSaveFileName(self, self.tr("Export image"), "", self.tr("Image (*.png)"))[0]
            image = cv2.imread(f'{self.pixmap_img}')   
            cv2.imwrite(f'{way_to_save}', image)     # way_to_save - путь куда сохраняем img
        except:
            pass
        
    def mouseReleaseEvent(self, event):  # вставка цвета пипетки
        try:
            if self.checkBox.isChecked() == False: 
                self.label_5.setText('Обнаружено колоний:')
                self.changeT(self.view.hsv_list)
        except:
            pass
        
############################# КЛАСС ПИПЕТКИ ################
        
class ScaleWindow(QWidget):
    # Увеличенное окно просмотра
    def __init__(self, *args, **kwargs):
        super(ScaleWindow, self).__init__(*args, **kwargs)
        self.setWindowFlags(Qt.Tool | 
                            Qt.FramelessWindowHint |
                            Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(1, 1)
        self.move(1, 1)
        self._image = None

    def updateImage(self, pos, image):
        self._image = image
        self.resize(image.size())
        self.move(pos.x() + 10, pos.y() + 10)
        self.show()
        self.update()

    def paintEvent(self, event):
        super(ScaleWindow, self).paintEvent(event)
        if self._image:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            path = QPainterPath()
            radius = min(self.width(), self.height()) / 2
            path.addRoundedRect(QRectF(self.rect()), radius, radius)
            painter.setClipPath(path)
            # изображение
            painter.drawImage(self.rect(), self._image)

            # Средний синий перекрестие
            painter.setPen(QPen(QColor(0, 174, 255), 3))
            hw = self.width() / 2
            hh = self.height() / 2
            painter.drawLines(
                QLineF(hw, 0, hw, self.height()),
                QLineF(0, hh, self.width(), hh)
            )

            # рамка
            painter.setPen(QPen(Qt.white, 3))
            painter.drawRoundedRect(self.rect(), radius, radius)


class CColorStraw(QPushButton):
    colorChanged = pyqtSignal(QColor)
    def __init__(self, parent):
        super(CColorStraw, self).__init__(parent)
        QFontDatabase.addApplicationFontFromData(QByteArray.fromBase64(FONT))
        font = self.font() or QFont()
        font.setFamily('iconfont')
        self.setFont(font)
        self.setToolTip('Нарисуйте цвет экрана')
        self._scaleWindow = ScaleWindow()
        # Не забудьте сначала показать его, а затем спрятать. 
        self._scaleWindow.show()
        self._scaleWindow.hide()

    def closeEvent(self, event):
        self._scaleWindow.close()
        super(CColorStraw, self).closeEvent(event)

    def mousePressEvent(self, event):
        super(CColorStraw, self).mousePressEvent(event)
        self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event):
        super(CColorStraw, self).mouseReleaseEvent(event)
        # Установите стиль мыши на нормальный
        self.setCursor(Qt.ArrowCursor)
        self._scaleWindow.hide()


    def mouseMoveEvent(self, event):
        super(CColorStraw, self).mouseMoveEvent(event)
        # Получить положение мыши на экране
        pos = event.globalPos()
        # Возьмите часть увеличенного изображения
        image = QApplication.primaryScreen().grabWindow(
            int(QApplication.desktop().winId()),
            pos.x() - 6, pos.y() - 6, 13, 13).toImage()
        color = image.pixelColor(6, 6)
        
        ####### Получаем цвет пипетки ################
        rgb_color = np.uint8([[[color.red(), color.green(), color.blue()]]])   
        self.hsv_list = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV).tolist()[0][0]
        self.hsv_list[0] = self.hsv_list[0] * 2 
      #  print(self.hsv_list)
        ####################
        
        
        if color.isValid():
            self.colorChanged.emit(color)
        self._scaleWindow.updateImage(pos, image.scaled(130, 130))

    
        
class MyWidget(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        
style_css = '''               
    QLabel, QCheckBox {
        color: rgb(255, 255, 255);   
    }
    QPushButton {
        border-radius: 7px;
    }
    QCheckBox::indicator {
    width: 20px;
    height: 20px;
    }
    
'''

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    app.setStyleSheet(style_css)
    ex.show()
    sys.exit(app.exec_())






