"""Tests for artifact and file uploads to production.

These tests verify that all file types (Artifact, File, Text, Image, Video, Audio)
can be properly uploaded to the production server without HTTP errors.
"""

import importlib.util
import json
import os
import tempfile

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage

import pluto
from tests.utils import get_task_name

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    HAS_TORCH = False

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:  # pragma: no cover - optional dependency
    sf = None
    HAS_SOUNDFILE = False

HAS_IMAGEIO = importlib.util.find_spec('imageio') is not None
HAS_MOVIEPY = importlib.util.find_spec('moviepy') is not None
HAS_VIDEO_DEPS = HAS_IMAGEIO and HAS_MOVIEPY

TESTING_PROJECT_NAME = 'testing-ci'


class TestArtifactUploads:
    """Test suite for artifact uploads to production."""

    def test_artifact_upload_from_file_path(self, tmp_path):
        """Test uploading an Artifact from a file path."""
        # Create a test file
        artifact_path = tmp_path / 'test_artifact.json'
        artifact_path.write_text(json.dumps({'test': 'data', 'value': 123}))

        artifact = pluto.Artifact(str(artifact_path), caption='json-artifact')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'artifact/json': artifact})
        run.finish()

    def test_artifact_upload_binary_file(self, tmp_path):
        """Test uploading a binary artifact."""
        # Create a binary file
        binary_path = tmp_path / 'test_binary.bin'
        binary_path.write_bytes(os.urandom(1024))  # 1KB random data

        artifact = pluto.Artifact(str(binary_path), caption='binary-artifact')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'artifact/binary': artifact})
        run.finish()

    def test_artifact_with_metadata(self, tmp_path):
        """Test uploading an artifact with metadata."""
        # Create a test file
        artifact_path = tmp_path / 'model_weights.pt'
        artifact_path.write_bytes(os.urandom(2048))

        artifact = pluto.Artifact(
            str(artifact_path),
            caption='model-weights',
            metadata={'version': '1.0', 'accuracy': 0.95, 'epochs': 100},
        )

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'artifact/model': artifact})
        run.finish()


class TestFileUploads:
    """Test suite for generic File uploads."""

    def test_file_upload_text_file(self, tmp_path):
        """Test uploading a generic text file using File class."""
        file_path = tmp_path / 'readme.txt'
        file_path.write_text('This is a test readme file.\nWith multiple lines.')

        file_obj = pluto.File(str(file_path), name='readme')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'file/readme': file_obj})
        run.finish()

    def test_file_upload_csv(self, tmp_path):
        """Test uploading a CSV file."""
        csv_path = tmp_path / 'data.csv'
        csv_path.write_text('col1,col2,col3\n1,2,3\n4,5,6\n7,8,9')

        file_obj = pluto.File(str(csv_path), name='data-csv')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'file/csv': file_obj})
        run.finish()

    def test_file_upload_large_file(self, tmp_path):
        """Test uploading a larger file (100KB)."""
        large_file_path = tmp_path / 'large_file.bin'
        large_file_path.write_bytes(os.urandom(100 * 1024))  # 100KB

        file_obj = pluto.File(str(large_file_path), name='large-file')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'file/large': file_obj})
        run.finish()


class TestTextUploads:
    """Test suite for Text file uploads."""

    def test_text_upload_from_string(self):
        """Test uploading Text from a string."""
        text_content = 'This is a test log message.\nWith multiple lines.\nAnd some data.'
        text = pluto.Text(text_content, caption='log-message')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'text/log': text})
        run.finish()

    def test_text_upload_from_file_path(self, tmp_path):
        """Test uploading Text from a file path."""
        text_path = tmp_path / 'log.txt'
        text_path.write_text('Log entry 1\nLog entry 2\nLog entry 3')

        text = pluto.Text(str(text_path), caption='log-file')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'text/file': text})
        run.finish()

    def test_text_upload_long_content(self):
        """Test uploading a longer text content."""
        # Generate a longer text content
        long_text = '\n'.join([f'Line {i}: ' + 'x' * 100 for i in range(100)])
        text = pluto.Text(long_text, caption='long-text')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'text/long': text})
        run.finish()

    def test_text_upload_unicode(self):
        """Test uploading text with unicode characters."""
        unicode_text = 'Hello World! \u4e2d\u6587 \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4 \U0001f600\U0001f680'
        text = pluto.Text(unicode_text, caption='unicode-text')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'text/unicode': text})
        run.finish()


class TestImageUploads:
    """Test suite for Image uploads."""

    def test_image_upload_from_file_path(self, tmp_path):
        """Test uploading an Image from a file path."""
        img_path = tmp_path / 'test_image.png'
        PILImage.new('RGB', (64, 64), color='red').save(img_path)

        image = pluto.Image(str(img_path), caption='file-path-image')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'image/file': image})
        run.finish()

    def test_image_upload_from_pil(self):
        """Test uploading an Image from a PIL Image object."""
        pil_img = PILImage.new('RGB', (64, 64), color='green')

        image = pluto.Image(pil_img, caption='pil-image')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'image/pil': image})
        run.finish()

    def test_image_upload_from_numpy(self):
        """Test uploading an Image from a numpy array."""
        np_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        image = pluto.Image(np_img, caption='numpy-image')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'image/numpy': image})
        run.finish()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not installed')
    def test_image_upload_from_matplotlib(self):
        """Test uploading an Image from a matplotlib figure."""
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1, 2, 3], [0, 1, 4, 9], 'b-')
        ax.set_title('Test Plot')

        image = pluto.Image(fig, caption='matplotlib-image')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'image/matplotlib': image})
        run.finish()
        plt.close(fig)

    @pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
    def test_image_upload_from_torch_tensor(self):
        """Test uploading an Image from a PyTorch tensor."""
        pytest.importorskip('torchvision.utils')
        tensor = torch.rand(3, 64, 64)

        image = pluto.Image(tensor, caption='torch-image')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'image/torch': image})
        run.finish()

    def test_image_upload_rgba(self):
        """Test uploading an RGBA image."""
        rgba_img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)

        image = pluto.Image(rgba_img, caption='rgba-image')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'image/rgba': image})
        run.finish()

    def test_image_upload_multiple(self, tmp_path):
        """Test uploading multiple images in a single log call."""
        images = []
        for i in range(3):
            img = PILImage.new('RGB', (32, 32), color=(i * 80, i * 80, 255 - i * 80))
            images.append(pluto.Image(img, caption=f'multi-image-{i}'))

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        for i, img in enumerate(images):
            run.log({f'image/multi/{i}': img})
        run.finish()


class TestVideoUploads:
    """Test suite for Video uploads."""

    def test_video_upload_from_file_path(self, tmp_path):
        """Test uploading a Video from a file path."""
        video_path = tmp_path / 'test_video.mp4'
        # Create a minimal valid-ish MP4 file (just bytes for testing upload)
        video_path.write_bytes(b'\x00' * 1024)

        video = pluto.Video(str(video_path), caption='file-video')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'video/file': video})
        run.finish()

    @pytest.mark.skipif(not HAS_VIDEO_DEPS, reason='video dependencies not installed')
    def test_video_upload_from_numpy(self):
        """Test uploading a Video from a numpy array."""
        # Shape: (batch, time, channels, height, width)
        video_array = np.random.randint(0, 255, (4, 3, 32, 32), dtype=np.uint8)

        video = pluto.Video(video_array, rate=10, caption='numpy-video')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'video/numpy': video})
        run.finish()

    @pytest.mark.skipif(
        not HAS_VIDEO_DEPS or not HAS_TORCH, reason='video or torch deps not installed'
    )
    def test_video_upload_from_torch_tensor(self):
        """Test uploading a Video from a PyTorch tensor."""
        # Shape: (time, channels, height, width)
        video_tensor = torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)

        video = pluto.Video(video_tensor, rate=10, caption='torch-video')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'video/torch': video})
        run.finish()

    def test_video_upload_gif_format(self, tmp_path):
        """Test uploading a video with GIF format."""
        gif_path = tmp_path / 'test.gif'
        # Create a minimal GIF
        gif_path.write_bytes(b'GIF89a\x01\x00\x01\x00\x00\x00\x00;\x00')

        video = pluto.Video(str(gif_path), format='gif', caption='gif-video')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'video/gif': video})
        run.finish()


class TestAudioUploads:
    """Test suite for Audio uploads."""

    def test_audio_upload_from_file_path(self, tmp_path):
        """Test uploading Audio from a file path."""
        audio_path = tmp_path / 'test_audio.wav'
        # Create a minimal WAV file with soundfile if available, otherwise raw bytes
        if HAS_SOUNDFILE:
            # Create a simple sine wave
            sample_rate = 22050
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            sf.write(str(audio_path), audio_data, sample_rate)
        else:
            # Write minimal WAV header + silence
            audio_path.write_bytes(
                b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00'
                b'\x01\x00\x01\x00"V\x00\x00"V\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00'
            )

        audio = pluto.Audio(str(audio_path), caption='file-audio')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'audio/file': audio})
        run.finish()

    @pytest.mark.skipif(not HAS_SOUNDFILE, reason='soundfile not installed')
    def test_audio_upload_from_numpy(self):
        """Test uploading Audio from a numpy array."""
        # Create a simple audio signal (1 second sine wave at 440Hz)
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio = pluto.Audio(audio_data, rate=sample_rate, caption='numpy-audio')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'audio/numpy': audio})
        run.finish()

    @pytest.mark.skipif(not HAS_SOUNDFILE, reason='soundfile not installed')
    def test_audio_upload_stereo(self):
        """Test uploading stereo audio."""
        sample_rate = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Stereo: left channel 440Hz, right channel 880Hz
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo_data = np.column_stack([left, right])

        audio = pluto.Audio(stereo_data, rate=sample_rate, caption='stereo-audio')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'audio/stereo': audio})
        run.finish()

    def test_audio_upload_from_url(self, tmp_path):
        """Test uploading audio downloaded from a URL."""
        url = 'https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg'
        audio_path = tmp_path / 'downloaded.ogg'

        try:
            response = httpx.get(url, timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network issues
            pytest.skip(f'Audio download failed: {exc}')

        audio_path.write_bytes(response.content)
        audio = pluto.Audio(str(audio_path), caption='url-audio')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'audio/url': audio})
        run.finish()


class TestMultipleFileTypesUpload:
    """Test uploading multiple file types in a single run."""

    def test_mixed_file_types_upload(self, tmp_path):
        """Test uploading multiple different file types in a single run."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )

        # Create and log a text file
        text = pluto.Text('Test log content', caption='mixed-text')
        run.log({'mixed/text': text})

        # Create and log an image
        img = PILImage.new('RGB', (32, 32), color='blue')
        image = pluto.Image(img, caption='mixed-image')
        run.log({'mixed/image': image})

        # Create and log an artifact
        artifact_path = tmp_path / 'config.json'
        artifact_path.write_text('{"setting": "value"}')
        artifact = pluto.Artifact(str(artifact_path), caption='mixed-artifact')
        run.log({'mixed/artifact': artifact})

        # Create and log a generic file
        file_path = tmp_path / 'data.txt'
        file_path.write_text('Some data content')
        file_obj = pluto.File(str(file_path), name='mixed-file')
        run.log({'mixed/file': file_obj})

        run.finish()

    def test_sequential_file_uploads(self, tmp_path):
        """Test uploading files sequentially across multiple steps."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )

        for step in range(5):
            # Log a metric
            run.log({'train/loss': 1.0 / (step + 1)}, step=step)

            # Log an image at each step
            img = PILImage.new(
                'RGB', (32, 32), color=(step * 50, 100, 255 - step * 50)
            )
            image = pluto.Image(img, caption=f'step-{step}-image')
            run.log({f'train/image': image}, step=step)

            # Log text at each step
            text = pluto.Text(f'Step {step} completed', caption=f'step-{step}-log')
            run.log({f'train/log': text}, step=step)

        run.finish()

    def test_batch_image_upload(self):
        """Test uploading a batch of images."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )

        # Create multiple images and log them in one call
        images = [
            pluto.Image(
                PILImage.new('RGB', (16, 16), color=(i * 25, i * 25, i * 25)),
                caption=f'batch-{i}',
            )
            for i in range(10)
        ]

        # Log images as a list
        run.log({'batch/images': images})
        run.finish()


class TestEdgeCases:
    """Test edge cases for file uploads."""

    def test_empty_text_upload(self):
        """Test uploading empty text."""
        text = pluto.Text('', caption='empty-text')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'edge/empty-text': text})
        run.finish()

    def test_small_image_upload(self):
        """Test uploading a very small image (1x1)."""
        img = PILImage.new('RGB', (1, 1), color='white')
        image = pluto.Image(img, caption='tiny-image')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'edge/tiny-image': image})
        run.finish()

    def test_large_image_upload(self):
        """Test uploading a larger image (512x512)."""
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = pluto.Image(img, caption='large-image')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'edge/large-image': image})
        run.finish()

    def test_special_characters_in_caption(self, tmp_path):
        """Test uploading files with special characters in captions."""
        # Create a file
        file_path = tmp_path / 'test.txt'
        file_path.write_text('test content')

        # Use a caption with special characters (which should be sanitized)
        artifact = pluto.Artifact(str(file_path), caption='test_artifact-v1.0')

        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )
        run.log({'edge/special-caption': artifact})
        run.finish()

    def test_multiple_files_same_key(self):
        """Test uploading multiple files with the same log key across steps."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME, name=get_task_name(), config={}
        )

        for step in range(3):
            img = PILImage.new('RGB', (16, 16), color=(step * 80, 0, 0))
            image = pluto.Image(img, caption=f'overwrite-{step}')
            run.log({'same-key/image': image}, step=step)

        run.finish()
