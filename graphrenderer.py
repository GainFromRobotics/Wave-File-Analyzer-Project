from abc import ABC, abstractmethod
from datamodel import AudioDataModel
from signalprocessor import SignalProcessor
from vispy import scene
from vispy.color import colormap
import numpy as np

# If possible, I think I can streamline how I make my classes
# Many of them are cut-paste. I need to re-write my plans for abstraction

#https://matplotlib.org/stable/users/explain/colors/colormaps.html
GRAPH_COLORMAP = 'plasma'
GRAPH_WINDOWSIZE = 1024
FFT_WINDOWSIZE = 4096
FFT_HOPSIZE = FFT_WINDOWSIZE // 2 # should be around 50% of FFT_WINDOWSIZE
GRAPH_LINE_WIDTH = 2

DB_MAX = 30
DB_MIN = -90

def _normalize_for_colormap(values, vmin, vmax):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

class VispyGraphRenderer(ABC):

    def __init__(self, graph_id: str, data_model: AudioDataModel):
        self.graph_id = graph_id
        self.data_model = data_model
        self.viewbox: scene.ViewBox | None = None
        self.axis_widgets = []
        self._axes_initialized = False

    def set_data_model(self, data_model: AudioDataModel):
        self.data_model = data_model

    #https://vispy.org/gallery/scene/axes_plot.html
    def _attach_axis_widgets(self,xaxis_label: str = '', yaxis_label:str=''):
        if self.viewbox is None:
            return

        if self._axes_initialized:
            return

        self.axis_widgets = []

        self.axis_widgets.append(scene.AxisWidget(
            orientation='left',
            axis_label=yaxis_label,
        ))

        self.axis_widgets.append(scene.AxisWidget(
                    orientation='bottom',
                    axis_label=xaxis_label
        ))

        for axis in self.axis_widgets:
            self.viewbox.add_widget(axis)
            axis.link_view(self.viewbox)

        #for orientation in ("left", "bottom"):
        #    axis = scene.AxisWidget(orientation=orientation)
        #    self.viewbox.add_widget(axis)
        #    axis.link_view(self.viewbox)
        #    self.axis_widgets.append(axis)

        self._axes_initialized = True

    def get_graph_id(self) -> str:
        return self.graph_id
    
    @abstractmethod
    def initialize_view(self, canvas: scene.SceneCanvas, grid: scene.Widget, row=0, col=0):
        """Create vispy view plus visuals"""
        pass

    @abstractmethod
    def update_view(self):
        """Update anything such as camera ranges or axes"""
        pass

    @abstractmethod
    def render_visual(self):
        """Called when data changes"""
        pass

class WaveformRenderer(VispyGraphRenderer):

    BORDER_COLOR = 'blue'

    def __init__(self, channel: int, data_model: AudioDataModel):
        super().__init__(f"waveform_{channel}", data_model)
        self.channel = channel

    def initialize_view(self, canvas: scene.SceneCanvas, grid: scene.Widget, row=0, col=0):
        #return super().initialize_view(grid, row, col)
        if self.viewbox is None:

            # Create Viewbox
            self.viewbox = scene.widgets.ViewBox(
                border_color=self.BORDER_COLOR, 
                parent=canvas.scene,
            )
            
            # Add view to canvas
            grid.add_widget(self.viewbox,row,col) #type: ignore

            # Create camera
            self.viewbox.camera = scene.PanZoomCamera()
            self.viewbox.camera.set_range(x=(0,GRAPH_WINDOWSIZE),y=(-1,1))

            # add axes to scene
            self._attach_axis_widgets(xaxis_label='Sample',yaxis_label='Normalized Value')

            # add line to scene
            self.line = scene.visuals.Line( #type: ignore
                pos=np.zeros((GRAPH_WINDOWSIZE,2),dtype=np.float32), 
                color='white', 
                width=2,
                parent=self.viewbox.scene
            )
        return
    
    def update_view(self):
        if self.viewbox is not None:
            self.viewbox.camera.set_range(x=(0,GRAPH_WINDOWSIZE),y=(-1,1))
            self.line.set_data(pos = np.zeros((GRAPH_WINDOWSIZE,2),dtype=np.float32))
        pass

    def render_visual(self):
        if self.data_model.pcm_samples is None:
            return
        
        current_frame = self.data_model.playback_state.current_frame
        start_frame = max(0,current_frame)
        nframes = GRAPH_WINDOWSIZE # arbitrary value for now

        samples = self.data_model.get_samples(self.channel, start_frame,nframes)

        if len(samples) == 0:
            return

        x = np.arange(len(samples))
        p = np.column_stack([x,samples]) # what's column stack?

        # change the line data
        self.line.set_data(pos = p)
        return



class SpectrogramRenderer(VispyGraphRenderer):

    BORDER_COLOR = 'cyan'

    def __init__(self, channel: int, data_model: AudioDataModel):
        super().__init__(f"spectrogram_{channel}", data_model)
        self.channel = channel
        self.window_size = FFT_WINDOWSIZE
        self.hop_size = FFT_HOPSIZE
        self.samples_per_pixel = 1 # decimation level (subject to change)
        self.tile_width = 2**13 # 8192
        self.tile_visuals = []

        self.spectrogram_width = 0.0
        self.spectrogram_freq_max = 0.0
        self.time_scale = 1.0
        self.freq_scale = 1.0

    def initialize_view(self, canvas: scene.SceneCanvas, grid: scene.Widget, row=0, col=0):
        #return super().initialize_view(grid, row, col)
        if self.viewbox is None:

            # Create Viewbox
            self.viewbox = scene.widgets.ViewBox(
                border_color=self.BORDER_COLOR, 
                parent=canvas.scene,
            )
            
            # Add view to canvas
            grid.add_widget(self.viewbox,row,col) #type: ignore

            # Create camera
            self.viewbox.camera = scene.PanZoomCamera()
            self.viewbox.camera.set_range(x=(0,GRAPH_WINDOWSIZE),y=(-1,1))

            #Add axes to scene
            self._attach_axis_widgets(xaxis_label='Time (s)',yaxis_label='Frequency (Hz)')
                
            # add seek line
            self.seek_line = scene.InfiniteLine(
                pos = 0,
                color=[1.0,1.0,0.6,1.0],
                parent=self.viewbox.scene
            )
            self.seek_line.order = 999 # generic value
        return
    
    def update_view(self):
        if self.data_model.pcm_samples is None:
            return
        
        self.seek_line.set_data(pos=0)
        
        spectrogram = self.data_model.get_spectrogram(self.channel,self.window_size,self.hop_size)
        
        if spectrogram is None:
            print('No spectrogram was computed')
            return

        if spectrogram.size == 0:
            print("empty spectrogram")
            return

        if spectrogram.shape[0] < 1 or spectrogram.shape[1] < 1:
            print("invalid spectrogram dimensions")
            return
        
        spectrogram = np.nan_to_num(spectrogram,nan=0.0,posinf=0.0,neginf=0.0)
        spectrogram = spectrogram.astype(np.float32,copy=False)

        # Update our axes
        sample_rate = self.data_model.metadata.sample_rate
        self.time_scale = self.hop_size / sample_rate
        if spectrogram.shape[0] > 1:
            self.freq_scale = (sample_rate / 2.0) / (spectrogram.shape[0] - 1)
        else:
            self.freq_scale = 1.0

        self.spectrogram_freq_max = self.freq_scale * (spectrogram.shape[0] - 1)

        # get max and  min values for our spectrogram
        DB_MAX = np.max(spectrogram)
        DB_MIN = np.min(spectrogram)

        for vis in self.tile_visuals:
            try:
                vis.parent = None
            except Exception:
                pass
        self.tile_visuals.clear()

        # Split spectrogram into
        total_width = 0
        for start in range(0, spectrogram.shape[1], self.tile_width):
            stop = min(start + self.tile_width, spectrogram.shape[1])
            tile = spectrogram[:,start:stop]

            if tile.size == 0:
                continue

            # each tile is small enough for a texture
            img = scene.visuals.Image( #type: ignore
                data=tile,
                cmap=GRAPH_COLORMAP,
                clim=(DB_MIN,DB_MAX),
                parent=self.viewbox.scene #type: ignore
            )
            img.order = 0
            img.transform = scene.transforms.STTransform(
                translate=(total_width * self.time_scale, 0, 0),
                scale=(self.time_scale, self.freq_scale, 1.0)
            )

            self.tile_visuals.append(img)
            total_width += tile.shape[1]

        self.spectrogram_width = total_width * self.time_scale

        # change camera range
        if self.viewbox is not None:
            self.viewbox.camera.set_range(
                x=(0, self.spectrogram_width),
                y=(0, self.spectrogram_freq_max)
            )
    
    def render_visual(self):

        if self.data_model.metadata.frame_count <= 0:
            return

        current_frame = self.data_model.playback_state.current_frame
        sample_rate = self.data_model.metadata.sample_rate
        seek_line_pos = current_frame / sample_rate
        self.seek_line.set_data(pos=seek_line_pos)



class ConstantQTransformRenderer(VispyGraphRenderer):

    BORDER_COLOR = 'red'

    def __init__(self, channel: int, data_model: AudioDataModel):
        super().__init__(f"CQT_{channel}", data_model)
        self.channel = channel
        self.window_size = 8192
        self.hop_size = 4096
        self.samples_per_pixel = 1 # decimation level (subject to change)
        self.tile_width = 2**13 # 8192
        self.tile_visuals = []

        self.CQT_width = 0.0
        #self.CQT_freq_max = 0.0
        self.time_scale = 1.0
        self.semitone_scale = 1.0

    def initialize_view(self, canvas: scene.SceneCanvas, grid: scene.Widget, row=0, col=0):
        #return super().initialize_view(grid, row, col)
        if self.viewbox is None:

            # Create Viewbox
            self.viewbox = scene.widgets.ViewBox(
                border_color=self.BORDER_COLOR, 
                parent=canvas.scene,
            )
            
            # Add view to canvas
            grid.add_widget(self.viewbox,row,col) #type: ignore

            # Create camera
            self.viewbox.camera = scene.PanZoomCamera()
            self.viewbox.camera.set_range(x=(0,GRAPH_WINDOWSIZE),y=(-1,1))

            #Add axes to scene
            self._attach_axis_widgets(xaxis_label='Time (s)',yaxis_label='Semitone')
                
            # add seek line
            self.seek_line = scene.InfiniteLine(
                pos = 0,
                color=[1.0,1.0,0.6,1.0],
                parent=self.viewbox.scene
            )
            self.seek_line.order = 999 # generic value
        return
    
    def update_view(self):
        if self.data_model.pcm_samples is None:
            return
        
        self.seek_line.set_data(pos=0)
        
        cqt = self.data_model.get_Constant_Q_Transform(self.channel,self.window_size,self.hop_size)
        
        if cqt is None:
            print('No cqt was computed')
            return

        if cqt.size == 0:
            print("empty cqt")
            return

        if cqt.shape[0] < 1 or cqt.shape[1] < 1:
            print("invalid cqt dimensions")
            return
        
        cqt = np.nan_to_num(cqt,nan=0.0,posinf=0.0,neginf=0.0)
        cqt = cqt.astype(np.float32,copy=False)

        # Update our axes
        sample_rate = self.data_model.metadata.sample_rate
        self.time_scale = self.hop_size / sample_rate

        semitone_min = SignalProcessor.semitones_A0
        semitone_max = SignalProcessor.semitones_A8

        if cqt.shape[0] > 1:
            self.semitone_scale = (semitone_max - semitone_min) / (cqt.shape[0] - 1)
        else:
            self.semitone_scale = 1.0

        #self.CQT_freq_min = semitone_min
        #self.CQT_freq_max = semitone_max

        # get max and  min values for our spectrogram
        self.lin_max = np.max(cqt)
        self.lin_min = np.min(cqt)

        for vis in self.tile_visuals:
            try:
                vis.parent = None
            except Exception:
                pass
        self.tile_visuals.clear()

        # Split CQT into
        total_width = 0
        for start in range(0, cqt.shape[1], self.tile_width):
            stop = min(start + self.tile_width, cqt.shape[1])
            tile = cqt[:,start:stop]

            if tile.size == 0:
                continue

            # each tile is small enough for a texture
            img = scene.visuals.Image( #type: ignore
                data=tile,
                cmap='viridis',
                clim=(self.lin_min,self.lin_max),
                parent=self.viewbox.scene #type: ignore
            )
            img.order = 0
            img.transform = scene.transforms.STTransform(
                translate=(total_width * self.time_scale, 0, 0),
                scale=(self.time_scale, self.semitone_scale, 1.0)
            )

            self.tile_visuals.append(img)
            total_width += tile.shape[1]

        self.CQT_width = total_width * self.time_scale

        # change camera range
        if self.viewbox is not None:
            self.viewbox.camera.set_range(
                x=(0, self.CQT_width),
                y=(semitone_min,semitone_max)
            )
    
    def render_visual(self):

        if self.data_model.metadata.frame_count <= 0:
            return

        current_frame = self.data_model.playback_state.current_frame
        sample_rate = self.data_model.metadata.sample_rate
        seek_line_pos = current_frame / sample_rate
        self.seek_line.set_data(pos=seek_line_pos)



class CepstrumRenderer(VispyGraphRenderer):

    BORDER_COLOR = 'green'
    CEPSTRUM_WINDOW_SIZE = 8192

    def __init__(self, channel: int, data_model: AudioDataModel):
        super().__init__(f"cepstrum_{channel}", data_model)
        self.channel = channel
        self.cm = colormap.MatplotlibColormap('viridis')

    def initialize_view(self, canvas: scene.SceneCanvas, grid: scene.Widget, row=0, col=0):
        #return super().initialize_view(grid, row, col)
        if self.viewbox is None:

            # Create Viewbox
            self.viewbox = scene.widgets.ViewBox(
                border_color=self.BORDER_COLOR, 
                parent=canvas.scene,
            )
            
            # Add view to canvas
            grid.add_widget(self.viewbox,row,col) #type: ignore

            # Create camera
            self.viewbox.camera = scene.PanZoomCamera()
            self.viewbox.camera.set_range(x=(0,self.CEPSTRUM_WINDOW_SIZE),y=(-1,1))

            # add axes to scene
            self._attach_axis_widgets(xaxis_label='Quefrency',yaxis_label='Energy')

            # add line to scene
            self.line = scene.visuals.Line( #type: ignore
                pos=np.zeros((self.CEPSTRUM_WINDOW_SIZE,2),dtype=np.float32), 
                color='white', 
                width=2,
                parent=self.viewbox.scene
            )
        return
    
    def update_view(self):
        if self.viewbox is not None:
            self.viewbox.camera.set_range(x=(0,self.CEPSTRUM_WINDOW_SIZE),y=(-0.02,0.1))
            self.line.set_data(pos = np.zeros((self.CEPSTRUM_WINDOW_SIZE,2),dtype=np.float32))
        pass

    def render_visual(self):
        if self.data_model.pcm_samples is None:
            return
        
        current_frame = self.data_model.playback_state.current_frame
        start_frame = max(0,current_frame)
        nframes = self.CEPSTRUM_WINDOW_SIZE # arbitrary value for now

        #samples = self.data_model.get_samples(self.channel, start_frame,nframes)
        cepstrum_samples = self.data_model.get_cepstrum(self.channel, start_frame, nframes,get_power_cepstrum=True)

        # Remove really high values
        cepstrum_samples = cepstrum_samples[:len(cepstrum_samples)//2]

        if len(cepstrum_samples) == 0:
            return

        x = np.arange(len(cepstrum_samples))
        p = np.column_stack([x,cepstrum_samples]) # what's column stack?

        norm = _normalize_for_colormap(cepstrum_samples, 0, 0.03) #unsure of these values
        col = self.cm[norm]

        # change the line data
        self.line.set_data(pos = p,color=col)
        return


class FFTRenderer(VispyGraphRenderer):

    BORDER_COLOR = 'orange'

    def __init__(self, channel: int, data_model: AudioDataModel):
        super().__init__(f"waveform_{channel}", data_model)
        self.channel = channel
        self.window_size = FFT_WINDOWSIZE
        self.graph_width = 128 # gonna get replaced anyways
        self.cm = colormap.MatplotlibColormap(GRAPH_COLORMAP)

    def initialize_view(self, canvas: scene.SceneCanvas, grid: scene.Widget, row=0, col=0):
        #return super().initialize_view(grid, row, col)
        if self.viewbox is None:

            # Create Viewbox
            self.viewbox = scene.widgets.ViewBox(
                border_color=self.BORDER_COLOR, 
                parent=canvas.scene,
            )
            
            # Add view to canvas
            grid.add_widget(self.viewbox,row,col) #type: ignore

            # Create camera
            self.viewbox.camera = scene.PanZoomCamera()
            self.viewbox.camera.set_range(x=(0,GRAPH_WINDOWSIZE),y=(DB_MIN,DB_MAX))

            # add axes to scene
            self._attach_axis_widgets(xaxis_label='Frequency (Hz)',yaxis_label='Volume (dB)')

            # add line to scene
            self.line = scene.visuals.Line( #type: ignore
                pos=np.zeros((GRAPH_WINDOWSIZE,2),dtype=np.float32), 
                color='white', 
                width=2,
                parent=self.viewbox.scene
            )
            # add color to the scene
            self.color = self.cm[np.zeros((GRAPH_WINDOWSIZE),dtype=np.float32)]
        return
    
    def update_view(self):
        if self.viewbox is not None:
            current_frame = self.data_model.playback_state.current_frame
            spectrum_decomposition = self.data_model.get_spectrum(self.channel,current_frame)
           
            self.viewbox.camera.set_range(x=(0,len(spectrum_decomposition)),y=(DB_MIN,DB_MAX))

            col = self.cm[np.zeros((GRAPH_WINDOWSIZE),dtype=np.float32)]

            self.line.set_data(
                pos = np.zeros((GRAPH_WINDOWSIZE,2),dtype=np.float32),
                color = col
            )
        pass

    def render_visual(self):
        if self.data_model.pcm_samples is None:
            return
        
        current_frame = self.data_model.playback_state.current_frame

        spectrum_decomposition = self.data_model.get_spectrum(self.channel,current_frame)

        if len(spectrum_decomposition) == 0:
            return

        x = np.arange(len(spectrum_decomposition))
        p = np.column_stack([x,spectrum_decomposition])

        # Now adjust the color based on intensity
        norm = _normalize_for_colormap(spectrum_decomposition, DB_MIN, DB_MAX)
        col = self.cm[norm]

        # change the line data
        self.line.set_data(pos = p, color = col)
        return



class HarmonicCombRenderer(VispyGraphRenderer):

    BORDER_COLOR = 'red'

    MIN_Y = 0.0
    MAX_Y = 50.0

    A0 = 27.5
    A8 = 7040.0

    MIN_X = SignalProcessor.freq_to_semitone(A0) # A0
    MAX_X = SignalProcessor.freq_to_semitone(A8) #A8

    def __init__(self, channel: int, data_model: AudioDataModel, max_notes: int = 16):
        super().__init__(f"note_identify_{channel}", data_model)
        self.channel = channel
        self.window_size = FFT_WINDOWSIZE
        self.max_notes = max_notes
        #self.graph_width = 128 # gonna get replaced anyways
        self.cm = colormap.MatplotlibColormap("winter")

    def initialize_view(self, canvas: scene.SceneCanvas, grid: scene.Widget, row=0, col=0):
        #return super().initialize_view(grid, row, col)
        if self.viewbox is None:

            # Create Viewbox
            self.viewbox = scene.widgets.ViewBox(
                border_color=self.BORDER_COLOR, 
                parent=canvas.scene,
            )
            
            # Add view to canvas
            grid.add_widget(self.viewbox,row,col) #type: ignore

            # Create camera
            self.viewbox.camera = scene.PanZoomCamera()
            self.viewbox.camera.set_range(x=(self.MIN_X,self.MAX_X),y=(self.MIN_Y,self.MAX_Y))

            # add axes to scene
            self._attach_axis_widgets(xaxis_label='Semitones from A4',yaxis_label='Confidence Score')

            # Add base line to scene
            self.base_line = scene.visuals.Line( #type: ignore
                pos = np.column_stack([np.asarray([self.MIN_X,self.MAX_X],dtype=np.float32),np.asarray([0,0],dtype=np.float32)]),
                color=[0.4,1.0,0.6,1.0],
                parent=self.viewbox.scene,
            )
            self.base_line.order = 999 # generic value

            # add lines to scene
            self.note_visuals = []
            for i in range(self.max_notes):
                line = scene.visuals.Line( #type:ignore
                    pos = np.zeros((2,2),dtype=np.float32),
                    color='blue', #for now
                    width=3,
                    parent=self.viewbox.scene
                )
                label = scene.visuals.Text( #type: ignore
                    "",
                    color='blue', #for now
                    font_size = 12,
                    anchor_x='center',
                    anchor_y='top',
                    parent=self.viewbox.scene    
                )
                #self.note_visuals.append((line,label))
                self.note_visuals.append({
                    "line": line,
                    "txt": label
                })
            # add color to the scene
            #self.color = self.cm[np.zeros((GRAPH_WINDOWSIZE),dtype=np.float32)]
        return
    
    def update_view(self):
        if self.viewbox is not None:
            #current_frame = self.data_model.playback_state.current_frame
            sample_rate = self.data_model.metadata.sample_rate

            self.viewbox.camera.set_range(x=(self.MIN_X,self.MAX_X),y=(self.MIN_Y,self.MAX_Y))
        pass

    def render_visual(self):
        if self.data_model.pcm_samples is None:
            return

        if not hasattr(self,'note_visuals'):
            return
        
        current_frame = self.data_model.playback_state.current_frame
        note_candidates = []
        #note_candidates = self.data_model.get_note_candidates(self.channel,current_frame,window_size=4096)

        if not note_candidates:
            self._clear_note_visuals()
            return

        for i,visual in enumerate(self.note_visuals):
            if i >= len(note_candidates):
                self._clear_note_visuals(i)
                continue

            note = note_candidates[i]
            note_freq = note["base_frequency"]
            note_score = note["final_score"]

            if not np.isfinite(note_freq) or not np.isfinite(note_score):
                self._clear_note_visuals(i)
                continue

            # Create our values for rendering
            x0 = SignalProcessor.freq_to_semitone(note_freq)
            y0 = note_score

            x = np.asarray([x0,x0])
            y = np.asarray([0,y0])
            p = np.column_stack([x,y]) # what's column stack?

            note_name, cents_off = SignalProcessor.get_note_and_difference(note_freq)

            vcol = _normalize_for_colormap(note_freq,self.A0,self.A8)
            color = self.cm[vcol]

            # Assign all our values
            visual["line"].set_data(pos = p,color=color)
            visual["txt"].pos = (x0,y0)
            visual["txt"].text = f"{note_name}+{cents_off}c"
            visual["txt"].color = color

    def _clear_note_visuals(self,i: int | None = None):

        if not hasattr(self,'note_visuals'):
            return

        if i is None:
            # clear all notes
            for visual in self.note_visuals:
                visual["line"].set_data(pos = np.zeros((2,2),dtype=np.float32),color=self.cm[0.0])
                visual["txt"].pos = (0, 0)
                visual["txt"].text = ""
                visual["txt"].color =self.cm[0.0]
        else:
            # clear only that visual
            visual = self.note_visuals[i]
            visual["line"].set_data(pos = np.zeros((2,2),dtype=np.float32))
            visual["txt"].pos = (0, 0)
            visual["txt"].text = ""