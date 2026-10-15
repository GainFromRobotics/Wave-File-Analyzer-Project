from typing import List

from PyQt6 import QtGui
from PyQt6.QtGui import QMouseEvent, QPalette

from PyQt6.QtCore import (
    QEvent, 
    QSize, 
    QTimer, 
    Qt,
    pyqtSignal, 
    qWarning,
    QIODevice
)

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
import numpy as np



# https://www.pythonguis.com/tutorials/custom-title-bar-pyqt6/
# Thanks Copilot for correcting the mistakes present in this on LINUX
class DarkModeTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.Highlight)
        self.initial_pos = None
        self.setMouseTracking(True)  # ensure we get move events if needed
        title_bar_layout = QHBoxLayout(self)
        title_bar_layout.setContentsMargins(1, 1, 1, 1)
        title_bar_layout.setSpacing(2)

        self.title = QLabel(f"{self.__class__.__name__}", self)
        self.title.setStyleSheet(
            """QLabel {
                   font-weight: bold;
                   border: 2px solid black;
                   border-radius: 12px;
                   margin: 2px;
                }
            """
        )
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if title := parent.windowTitle():
            self.title.setText(title)
        title_bar_layout.addWidget(self.title)

        # Min button
        self.min_button = QToolButton(self)
        min_icon = self.style().standardIcon( # type: ignore
            QStyle.StandardPixmap.SP_TitleBarMinButton,
        )
        self.min_button.setIcon(min_icon)
        self.min_button.clicked.connect(self.window().showMinimized) # type: ignore

        # Max button
        self.max_button = QToolButton(self)
        max_icon = self.style().standardIcon( # type: ignore
            QStyle.StandardPixmap.SP_TitleBarMaxButton,
        )
        self.max_button.setIcon(max_icon)
        self.max_button.clicked.connect(self.window().showMaximized) # type: ignore

        # Close button
        self.close_button = QToolButton(self)
        close_icon = self.style().standardIcon( # type: ignore
            QStyle.StandardPixmap.SP_TitleBarCloseButton
        )
        self.close_button.setIcon(close_icon)
        self.close_button.clicked.connect(self.window().close) # type: ignore

        # Normal button
        self.normal_button = QToolButton(self)
        normal_icon = self.style().standardIcon( # type: ignore
            QStyle.StandardPixmap.SP_TitleBarNormalButton
        )
        self.normal_button.setIcon(normal_icon)
        self.normal_button.clicked.connect(self.window().showNormal) # type: ignore
        self.normal_button.setVisible(False)
        # Add buttons
        buttons = [
            self.min_button,
            self.normal_button,
            self.max_button,
            self.close_button,
        ]
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(QSize(28, 28))
            button.setStyleSheet(
                """QToolButton {
                        border: 2px solid white;
                        border-radius: 12px;
                    }
                """
            )
            title_bar_layout.addWidget(button)

    def window_state_changed(self, state):
        if state == Qt.WindowState.WindowMaximized:
            self.normal_button.setVisible(True)
            self.max_button.setVisible(False)
        else:
            self.normal_button.setVisible(False)
            self.max_button.setVisible(True)

    def mousePressEvent(self, a0:QtGui.QMouseEvent|None):
        if a0 is None:
            return
        if a0.button() == Qt.MouseButton.LeftButton:
            # if user clicked a control (min/max/close) let it handle the event
            clicked = self.childAt(a0.pos())
            from PyQt6.QtWidgets import QToolButton
            if isinstance(clicked, QToolButton):
                return super().mousePressEvent(a0)

            # Preferred: use the platform/system move (lets WM handle dragging)
            wh = self.window().windowHandle() #type: ignore
            try:
                if wh is not None and hasattr(wh, "startSystemMove"):
                    wh.startSystemMove()
                    a0.accept()
                    return
            except Exception:
                # fall back to manual dragging if system move not available
                pass

            # Fallback manual drag (no grabMouse)
            self._drag_mouse_pos = a0.globalPosition().toPoint()
            self._drag_window_pos = self.window().pos()  # type: ignore
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            a0.accept()
            return
        super().mousePressEvent(a0)

    def mouseMoveEvent(self, a0:QtGui.QMouseEvent|None):
        if a0 is None:
            return
        if getattr(self, "_drag_mouse_pos", None) is not None:
            current_global = a0.globalPosition().toPoint()
            delta = current_global - self._drag_mouse_pos  # type: ignore
            new_x = self._drag_window_pos.x() + delta.x()  # type: ignore
            new_y = self._drag_window_pos.y() + delta.y()  # type: ignore
            if not self.window().isMaximized():  # type: ignore
                self.window().move(new_x, new_y)  # type: ignore
            a0.accept()
            return
        super().mouseMoveEvent(a0)

    def mouseReleaseEvent(self, a0:QtGui.QMouseEvent|None):
        if a0 is None:
            return
        # Clear drag state
        self._drag_mouse_pos = None
        self._drag_window_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(a0)
        a0.accept()



def convert_to_units(a:int,y:List[int]) -> List[int]:
    x: List[int] = []
    
    b = a
    for yi in y:
        if yi <= 0:
            raise ValueError("Unit conversions must be a positive value")
        (q,r) = divmod(b,yi)
        x.append(q)
        b = r

    return x

def frames_to_readable_time_ms(frame: int, sample_rate: int) -> str:

    if frame < 0:
        raise ValueError("Cannot convert negative value to a time")
    
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive!")
    
    total_ms: float = frame / sample_rate * 1000

    MS_PER_SEC: float = 1000 # milliseconds per second
    SEC_PER_MIN: float = 60 # seconds per minute
    MIN_PER_HR: float = 60 # minutes per hour

    # FRAME_PER_MS * MS_PER_SEC = FRAMES_PER_SEC, which is our sample rate
    y: List[int] = [
            MS_PER_SEC*SEC_PER_MIN*MIN_PER_HR, # ms per hour
            MS_PER_SEC*SEC_PER_MIN, # ms per minute
            MS_PER_SEC, # ms per second
            1
        ]
    
    # floor our millisecond value
    x = convert_to_units(int(total_ms),y)

    hours = x[0]
    minutes = x[1]
    seconds = x[2]
    milliseconds = x[3]

    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}:{int(milliseconds):03d}"

def frames_to_readable_time_i(frame: int, sample_rate: int) -> str:
        
    if frame < 0:
        raise ValueError("Cannot convert negative value to a time")
    
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive!")

    SEC_PER_MIN: int = 60 # seconds per minute
    MIN_PER_HR: int = 60 # minutes per hour

    # FRAME_PER_MS * MS_PER_SEC = FRAMES_PER_SEC, which is our sample rate
    y: List[int] = [
            sample_rate*SEC_PER_MIN*MIN_PER_HR, # frames per hour
            sample_rate*SEC_PER_MIN, # frames per minute
            sample_rate, # frames per second
            1
        ]

    x = convert_to_units(frame,y)

    hours = x[0]
    minutes = x[1]
    seconds = x[2]
    frames = x[3]

    sample_rate_length = len(str(sample_rate))
    print(sample_rate_length)

    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d},i{str(frames).rjust(sample_rate_length,'0')}"



class TimerPlaybackWidget(QWidget):

    signal_timerManuallyChanged = pyqtSignal() # When the timer is changed manually
    
    def __init__(self, parent: QWidget, sample_rate: int) -> None:
        super().__init__(parent)

        self.sample_rate = sample_rate
        self.min_time_frames = 0
        self.max_time_frames = 0
        self.time_internal_frames = 0 # current time
        self.time_target_frames = 0 # target time
        self._frame_step = max(1, self.sample_rate // 20)
        self._mouse_threshold = 20
        self._mouse_start_pos = None
        self._mouse_dx = 0
        self._keyboard_direction = 0

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMinimumSize(220, 60)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #20232a;
                color: #ffffff;
                border: 2px solid #888888;
                border-radius: 10px;
            }
            """
        )

        self.label_time = QLabel(frames_to_readable_time_ms(0,10), self)
        self.label_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_time.setStyleSheet("font-size: 16px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.label_time)
        self.setLayout(layout)

        self.timer_user_input = QTimer(self) # rate-limit our key inputs
        self.timer_user_input.setInterval(80)
        self.timer_user_input.timeout.connect(self._keyboard_step)
        self.timer_user_input.timeout.connect(self._mouse_step)

    def set_timer_range(self, new_min, new_max):
        self.min_time_frames = new_min
        self.max_time_frames = new_max

        #print(f"max:{self.max_time_frames},min:{self.min_time_frames}")

    def set_sample_rate(self, new_sample_rate):
        self.sample_rate = new_sample_rate
        self._frame_step = max(1, self.sample_rate // 20)

    def set_readable_time(self, new_time_frames):
        self.time_internal_frames = self._clamp_frames(new_time_frames)
        curr_time = frames_to_readable_time_ms(self.time_internal_frames, self.sample_rate)
        max_time = frames_to_readable_time_ms(self.max_time_frames,self.sample_rate)
        self.label_time.setText(f"{curr_time} / {max_time}")

    def get_target_time(self):
        return self.time_target_frames

    def get_internal_time(self):
        return self.time_internal_frames

    def _clamp_frames(self, frames: int) -> int:
        return max(self.min_time_frames, min(self.max_time_frames, frames))

    def _adjust_time(self, delta_frames: int):
        self.time_target_frames = self.time_internal_frames + delta_frames
        self.set_readable_time(self.time_internal_frames + delta_frames)

    def _keyboard_step(self):
        if self._keyboard_direction != 0:
            self._adjust_time(self._keyboard_direction * self._frame_step)
            self.signal_timerManuallyChanged.emit()

    def _mouse_step(self):
        if self._mouse_dx != 0:
            self._adjust_time(int(self._mouse_dx*self._mouse_frame_step_multiple))
            self.signal_timerManuallyChanged.emit()

    # Highlight
    def enterEvent(self, event):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #20232a;
                color: #ffffff;
                border: 2px solid #61afef;
                border-radius: 10px;
            }
            """
        )
        self.setFocus()
        super().enterEvent(event)

    # Undo highlight
    def leaveEvent(self, event): # type: ignore
        self.setStyleSheet(
            """
            QWidget {
                background-color: #20232a;
                color: #ffffff;
                border: 2px solid #888888;
                border-radius: 10px;
            }
            """
        )
        super().leaveEvent(event)

    _mouse_frame_step_multiple_FAST = 60
    _mouse_frame_step_multiple_SLOW = 10
    _mouse_frame_step_multiple_DEFAULT = 20

    _mouse_frame_step_multiple = _mouse_frame_step_multiple_DEFAULT

    def mousePressEvent(self, event: QMouseEvent): # type: ignore
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._mouse_start_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.timer_user_input.start()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent): #type: ignore
        if self._mouse_start_pos is None:
            return super().mouseMoveEvent(event)

        dx = event.position().toPoint().x() - self._mouse_start_pos.x()

        if abs(dx) > self._mouse_threshold:
            self._mouse_dx = dx
        else:
            self._mouse_dx = 0

        event.accept()
        self.signal_timerManuallyChanged.emit()

    def mouseReleaseEvent(self, event): # type: ignore
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._mouse_start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.timer_user_input.stop()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    _frame_step_multiple_FAST = 3
    _frame_step_multiple_SLOW = 0.5
    _frame_step_multiple_DEFAULT = 1

    _frame_step_multiple = 1

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None: #type: ignore
        if event is None:
            return super().keyPressEvent(event)
        
        # Speed modifiers
        if event.key() == Qt.Key.Key_Shift:
            self._frame_step_multiple = self._frame_step_multiple_FAST
            self._mouse_frame_step_multiple = self._mouse_frame_step_multiple_FAST
        if event.key() == Qt.Key.Key_Control:
            self._frame_step_multiple = self._frame_step_multiple_SLOW
            self._mouse_frame_step_multiple = self._mouse_frame_step_multiple_SLOW

        frame_step = int(self._frame_step * self._frame_step_multiple)

        # Lef - Right Control
        if event.key() == Qt.Key.Key_Left and self._keyboard_direction != -1:
            self._keyboard_direction = -1
            #self._adjust_time(-frame_step)
            self.timer_user_input.start()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right and self._keyboard_direction != 1:
            self._keyboard_direction = 1
            #self._adjust_time(frame_step)
            self.timer_user_input.start()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent): # type: ignore

        # Speed modifiers (return to default)
        if event.key() == Qt.Key.Key_Shift:
            self._frame_step_multiple = self._frame_step_multiple_DEFAULT
            self._mouse_frame_step_multiple = self._mouse_frame_step_multiple_DEFAULT
        if event.key() == Qt.Key.Key_Control:
            self._frame_step_multiple = self._frame_step_multiple_DEFAULT
            self._mouse_frame_step_multiple = self._mouse_frame_step_multiple_DEFAULT

        # Left - Right Control
        if event.key() == Qt.Key.Key_Left and self._keyboard_direction == -1:
            self._keyboard_direction = 0
            self.timer_user_input.stop()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right and self._keyboard_direction == 1:
            self._keyboard_direction = 0
            self.timer_user_input.stop()
            event.accept()
            return
        super().keyReleaseEvent(event)