import sys
from typing import List

from datamodel import AudioDataModel, PlaybackState
from signalprocessor import SignalProcessor
from graphrenderer import (
    VispyGraphRenderer, 
    WaveformRenderer, 
    CepstrumRenderer,
    SpectrogramRenderer,
    ConstantQTransformRenderer,
    FFTRenderer,
    HarmonicCombRenderer,
)
from guiassets import DarkModeTitleBar, TimerPlaybackWidget

import numpy as np

from PyQt6.QtCore import (
    QEvent, 
    QSize, 
    QTimer, 
    Qt, 
    qWarning,
    QIODevice
)
from PyQt6.QtMultimedia import (
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
    QAudio,
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

from PyQt6.QtGui import QFont, QFontDatabase
QT_VER = 6


import vispy
from vispy import scene
from vispy.color import Color
from vispy.scene import Grid



class AudioSourceIODevice(QIODevice):

    def __init__(self, data:bytes, parent=None):
        super().__init__(parent)
        self._data = data
        self._pos = 0

    def open(self, mode=QIODevice.OpenModeFlag.ReadOnly):
        return super().open(mode)
    
    def bytesAvailable(self) -> int:
        return len(self._data) - self._pos + super().bytesAvailable()
    
    def readData(self, maxlen: int):
        if self._pos >= len(self._data):
            return b''
        
        #TODO: Ensure this is protected!
        chunk = self._data[self._pos: self._pos + maxlen]
        self._pos += len(chunk)
        return chunk
    
    # Not using this
    def writeData(self, a0):
        return -1
    
    def seek(self, pos: int):
        if 0 <= pos <= len(self._data):
            self._pos = pos
            return True
        return False


FPS_60_MS = int(1000/60)
FPS_100_MS = int(1000/100)

class AudioVisualizerApp(QMainWindow):

    TITLE = "Audio Visualizer v0.0.3"

    def __init__(self,devices:List[QAudioDevice]):
        super().__init__()
        self.devices = devices
        self.device = self.devices[0]

        #self.audioSink = None
        self.audioSource = None
        #self.format = None

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        self.setWindowTitle(self.TITLE)
        self.setGeometry(100,100,1600,1000) #silly values for now

        # Core components
        self.data_model = AudioDataModel()

        # When data is changed, we announce this to every system that uses that data
        self.data_model.data_changed.connect(self.event_dataModelChanged)

        # Create UI
        self.setup_ui()

        # Create Draw Timer
        self.timer_draw = QTimer()
        self.timer_draw.timeout.connect(self.render_visualizers)
        self.timer_draw.start(FPS_100_MS)

        # FPS tracking
        self._fps_count = 0
        self.timer_fps = QTimer()
        self.timer_fps.timeout.connect(self.update_fps_display)
        self.timer_fps.start(1000)

        # Set our playback state
        self.playback_state: QAudio.State = QAudio.State.StoppedState

    # Creates our graphs and control layout
    def setup_ui(self):
        central_widget : QWidget = QWidget()

        # Create central widget layout
        layout_central_widget = QVBoxLayout(central_widget)
        layout_central_widget.setContentsMargins(0, 0, 0, 0)
        layout_central_widget.setSpacing(0)

        # Create dark mode title bar
        self.title_bar = DarkModeTitleBar(self)
        # Add title bar to central widget layout
        layout_central_widget.addWidget(self.title_bar)
        
        # Create main canvas
        self.canvas: scene.SceneCanvas = scene.SceneCanvas(keys='interactive',title=self.TITLE,show=True)
        # Add canvas to our central widget layout
        layout_central_widget.addWidget(self.canvas.native, stretch=1)

        # Add control layout to central widget layout
        self.create_control_layout(layout_central_widget)

        # Add a grid to our canvas. This will be used for our visualizers
        self.grid_visualizer = self.canvas.central_widget.add_grid()
        self.grid_visualizer.padding = 20
        # Add our visualizers to the grid
        self.create_visualizers(self.grid_visualizer)
        
        # Apply layout to central widget
        central_widget.setLayout(layout_central_widget)
        # Assign our central widget
        self.setCentralWidget(central_widget)

        # Display our canvas
        self.canvas.show()
        self.canvas.update()

    def create_control_layout(self, parent_layout: QVBoxLayout):

        # Control Panel layout
        layout_control = QHBoxLayout()

        # Load WAV File button
        self.btn_load = QPushButton("Load Wav File")
        self.btn_load.clicked.connect(self.load_audio_file)
        layout_control.addWidget(self.btn_load)

        # Playback Timer Visual
        self.widget_playbackTimer = TimerPlaybackWidget(self,44100)
        self.widget_playbackTimer.signal_timerManuallyChanged.connect(self.event_timerManuallyChanged)
        layout_control.addWidget(self.widget_playbackTimer)

        # Play/Pause Button
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_playback)
        layout_control.addWidget(self.btn_play)

        # Volume Control
        label_volume = QLabel("Volume:")
        self.slider_volume = QSlider(Qt.Orientation.Horizontal)
        self.slider_volume.sliderMoved.connect(self.event_volumeChanged)
        layout_control.addWidget(label_volume)
        layout_control.addWidget(self.slider_volume)

        # Audio Output dropdown menu
        self.combobox_device_output = QComboBox()
        self.combobox_device_output.activated[int].connect(self.event_deviceChanged)
        for deviceInfo in self.devices:
            self.combobox_device_output.addItem(deviceInfo.description(), deviceInfo)
        layout_control.addWidget(self.combobox_device_output)

        # Add FPS display
        self.label_fps = QLabel("FPS: 0")
        layout_control.addWidget(self.label_fps)

        # Add control layout to parent widget
        parent_layout.addLayout(layout_control)

    def create_visualizers(self,grid):
        self.renderers: List[VispyGraphRenderer] = []

        '''if self.data_model.metadata and self.data_model.metadata.nchannels >=2:
            channels = self.data_model.metadata.nchannels
        else:
            channels = 1'''
        
        channels = 2

        # Create a renderer for each channel
        for ch in range(min(channels,2)):

            # Spectrogram Viewer
            spectrogram = SpectrogramRenderer(ch, self.data_model)
            spectrogram.initialize_view(self.canvas,grid,ch,0)
            self.renderers.append(spectrogram)

            # Waveform Viewer
            waveform = WaveformRenderer(ch, self.data_model)
            waveform.initialize_view(self.canvas,grid,ch,1)
            self.renderers.append(waveform)

            # CQT Viewer (Slow AF)
            #cqt = ConstantQTransformRenderer(ch,self.data_model)
            #cqt.initialize_view(self.canvas,grid,ch,2)
            #self.renderers.append(cqt)

            # Cepstrum Viewer
            cepstrum = CepstrumRenderer(ch, self.data_model)
            cepstrum.initialize_view(self.canvas,grid,ch,2)
            self.renderers.append(cepstrum)
            
            # FTgraph View
            #ftgraph = FFTRenderer(ch,self.data_model)
            #ftgraph.initialize_view(self.canvas,grid,ch,2)
            #self.renderers.append(ftgraph)

            # Note Identifier
            #noteidentifier = NoteIdentifierRenderer(ch,self.data_model)
            #noteidentifier.initialize_view(self.canvas,grid,ch,2)
            #self.renderers.append(noteidentifier)

        # Link cameras for renderers of the same type
        renderer_types = {
            SpectrogramRenderer: [],
            ConstantQTransformRenderer: [],
            WaveformRenderer: [],
            CepstrumRenderer: [],
            FFTRenderer: [],
            HarmonicCombRenderer: [],
        }
        
        for renderer in self.renderers:
            renderer_type = type(renderer)
            if renderer_type in renderer_types:
                renderer_types[renderer_type].append(renderer)
        
        # For each renderer type, link all cameras to the master
        for renderer_type, renderers_of_type in renderer_types.items():
            if renderers_of_type:
                master_camera = renderers_of_type[0].viewbox.camera
                for renderer in renderers_of_type[1:]:
                    renderer.viewbox.camera.link(master_camera)

        for renderer in self.renderers:
            print(renderer)

    # Initialize our Audio System
    def init_audio(self, samplerate:int, nchannels:int, samplewidth:int):

        self.format = QAudioFormat()
        self.format.setSampleRate(samplerate)
        self.format.setChannelCount(nchannels)

        match samplewidth:
            case 1:
                self.format.setSampleFormat(QAudioFormat.SampleFormat.UInt8)
            case 2:
                self.format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            case 4:
                self.format.setSampleFormat(QAudioFormat.SampleFormat.Int32)
            case _:
                self.format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        
        self.create_audio_output()

    def create_audio_output(self):
        if getattr(self, "audioSink", None) is not None:
            self.audioSink.stop()
            try:
                self.audioSink.stateChanged.disconnect(self.event_audiosinkStateChanged)
            except Exception:
                pass

        self.audioSink = QAudioSink(self.device, self.format)
        self.audioSink.stateChanged.connect(self.event_audiosinkStateChanged)

    """
    This is the function called by our loop timer.
    This function is responsible for running all the visual updates nessecary per GUI update
    """
    def render_visualizers(self):

        #================================
        # UPDATE FPS
        #================================
        try:
            self._fps_count += 1
        except Exception:
            self._fps_count = 0

        if self.playback_state == QAudio.State.ActiveState and self.audioSource is not None:

            #print(self.data_model.playback_state.current_frame)

            #================================
            # UPDATE PLAYBACK STATE AND TIMER
            #================================

            bytes_per_frame = self.data_model.frame_to_byte_offset(1)
            if bytes_per_frame > 0:
                current_frame = int(self.audioSource.pos() // bytes_per_frame)
            else:
                current_frame = self.data_model.playback_state.current_frame

            self.data_model.set_playback_state(QAudio.State.ActiveState, current_frame)

            total = self.data_model.playback_state.duration_frames
            if total > 0:
                self.widget_playbackTimer.set_readable_time(current_frame)

            #================================
            # UPDATE GRAPHS
            #================================

            for renderer in self.renderers:
                renderer.render_visual()
                #print(f"{renderer}.update_data() called")
            self.canvas.update()

    def update_fps_display(self):
        try:
            fps = int(self._fps_count)
        except Exception:
            fps = 0
        self._fps_count = 0
        if getattr(self, "label_fps", None) is not None:
            self.label_fps.setText(f"FPS: {fps}")

    # FIXME: playback frame is not properly being assigned after the play button is pressed!
    def event_timerManuallyChanged(self):
        pass
        timer_displayed_frame = self.widget_playbackTimer.get_internal_time()
        '''timer_displayed_frame = self.widget_playbackTimer.get_internal_time()

        if getattr(self, "data_model", None) is None:
            return

        if getattr(self,"audioSink",None) is None:
            return

        frame_count = self.data_model.metadata.frame_count
        timer_displayed_frame = max(0,min(timer_displayed_frame,frame_count))

        # Just change the current frame
        self.data_model.set_playback_state(
            self.data_model.playback_state.status,
            timer_displayed_frame
        )

        # If there's no audio source, make one
        if self.audioSource is None:
            self.audioSource = AudioSourceIODevice(self.data_model.get_pcm_bytes())
            self.audioSource.open(QIODevice.OpenModeFlag.ReadOnly)
        else:
            byte_offset = self.data_model.frame_to_byte_offset(timer_displayed_frame)
            if byte_offset >= 0:
                self.audioSource.seek(byte_offset)

        # stop and start
        if self.audioSink.state() == QAudio.State.ActiveState:
            self.audioSink.stop()
            self.audioSink.start(self.audioSource)

        # Render graphics
        for renderer in self.renderers:
            renderer.render_visual()
        self.canvas.update()

        print(self.data_model.playback_state.current_frame)'''

    def event_dataModelChanged(self):
        print("DATA MODEL CHANGED")
        for renderer in self.renderers:
            renderer.set_data_model(self.data_model)
            renderer.update_view()
        self.canvas.update()

    def event_audiosinkStateChanged(self, state):
        self.playback_state = state

    def event_deviceChanged(self, index: int):
        #FIXME: add "stop"
        self.device = self.combobox_device_output.itemData(index)

    # What is this in dB relative to my output?
    def event_volumeChanged(self, value):
        if getattr(self, "audioSink", None) is not None:
            self.audioSink.setVolume(value / 100)
        else:
            qWarning("No Audio loaded to change volume for")

    def load_audio_file(self):
        filedialog = QFileDialog(self,"Open WAV File")
        filedialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        filedialog.setViewMode(QFileDialog.ViewMode.Detail)
        filedialog.setNameFilter("Wave Files (*.wav)")
        filedialog.setDefaultSuffix("wav")
        filedialog.setOptions(QFileDialog.Option.ReadOnly)

        if not filedialog.exec():
            return

        filenames = filedialog.selectedFiles()
        if not filenames:
            return

        filename = filenames[0]
        if not filename.lower().endswith(".wav"):
            QMessageBox.warning(
                self,
                "Invalid File",
                "Please select a WAV file (*.wav)."
            )
            return
        
        # This function also emits the data_changed signal
        self.data_model.load_audio(filename)

        # reset playback / source state for the newly loaded file
        self.playback_state = QAudio.State.StoppedState
        if getattr(self, "audioSink", None) is not None:
            self.audioSink.stop()
        self.audioSource = None

        sample_rate = getattr(self.data_model.metadata, "sample_rate",44100)
        sample_width = getattr(self.data_model.metadata, "sample_width",2)
        nchannels = getattr(self.data_model.metadata, "nchannels", 1)

        frame_count = getattr(self.data_model.metadata, "frame_count", 44100*60)

        self.init_audio(sample_rate, nchannels, sample_width)
        self.btn_play.setEnabled(True) # NOW we can use the play button
        self.slider_volume.setValue(100) # assume volume is all the way up
        self.widget_playbackTimer.set_sample_rate(sample_rate)
        self.widget_playbackTimer.set_timer_range(0,frame_count) # change playback timer range
        self.widget_playbackTimer.set_readable_time(0)
    
    def toggle_playback(self):

        if getattr(self, "audioSink", None) is None:
            return
        
        if self.audioSource is None:
            self.audioSource = AudioSourceIODevice(self.data_model.get_pcm_bytes())
            self.audioSource.open(QIODevice.OpenModeFlag.ReadOnly)

        match self.audioSink.state():
            case QAudio.State.SuspendedState: #paused -> playing
                self.btn_play.setText("Pause")
                self.data_model.set_playback_state(QAudio.State.ActiveState,self.data_model.playback_state.current_frame)
                self.audioSink.resume()
                qWarning("status: Suspended, resume()")
                
            case QAudio.State.ActiveState: #playing -> paused
                self.btn_play.setText("Play")
                self.data_model.set_playback_state(QAudio.State.SuspendedState,self.data_model.playback_state.current_frame)
                self.audioSink.suspend()
                qWarning("status: Active, suspend()")

            case QAudio.State.StoppedState: #stopped -> playing
                self.btn_play.setText("Pause")
                self.data_model.set_playback_state(QAudio.State.ActiveState,self.data_model.playback_state.current_frame)
                #self.audioSource.seek(self.data_model.frame_to_byte_offset(self.data_model.playback_state.current_frame))
                self.audioSink.start(self.audioSource)
                qWarning("status: Suspended, resume()")

            case QAudio.State.IdleState: #idle, treat like stopped. 
                self.btn_play.setText("Idle")
                self.data_model.set_playback_state(QAudio.State.ActiveState,self.data_model.playback_state.current_frame)
                #self.audioSource.seek(self.data_model.frame_to_byte_offset(self.data_model.playback_state.current_frame))
                self.audioSink.start(self.audioSource)
                qWarning("status: Stop")



if __name__ == '__main__':

    #print(vispy.sys_info()) # I'm curious

    font_id = QFontDatabase.addApplicationFont("")

    app = QApplication(sys.argv)

    audio_output_devices = QMediaDevices.audioOutputs()
    if not audio_output_devices:
        print('No Audio Output Found',file=sys.stderr)
        sys.exit(-1)

    window = AudioVisualizerApp(audio_output_devices)
    window.show()

    with open("style.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)

    sys.exit(app.exec())