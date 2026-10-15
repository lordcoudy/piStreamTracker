"""Interpreter discovery without installed inference runtimes or model downloads."""

import builtins
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pistream.detect import PoseDetector


RUNTIMES = (
    'tflite_runtime.interpreter',
    'ai_edge_litert.interpreter',
    'tensorflow.lite.python.interpreter',
)


class DetectorRuntimeTests(unittest.TestCase):
    def test_uses_first_available_runtime(self):
        original_import = builtins.__import__
        for selected in range(len(RUNTIMES)):
            with self.subTest(runtime=RUNTIMES[selected]):
                interpreter = Mock()
                interpreter.get_input_details.return_value = [{
                    'shape': (1, 192, 192, 3), 'dtype': np.uint8, 'index': 0,
                }]
                interpreter.get_output_details.return_value = [{'index': 1}]
                factory = Mock(return_value=interpreter)
                attempts = []

                def import_runtime(name, *args, **kwargs):
                    if name in RUNTIMES:
                        attempts.append(name)
                        if name != RUNTIMES[selected]:
                            raise ImportError(name)
                        return SimpleNamespace(Interpreter=factory)
                    return original_import(name, *args, **kwargs)

                with patch('builtins.__import__', side_effect=import_runtime), \
                        patch.object(PoseDetector, '_get_model', return_value='/cached/model.tflite'), \
                        patch('pistream.detect.urllib.request.urlopen') as download:
                    detector = PoseDetector(threads=3)
                self.assertTrue(detector.ready)
                self.assertEqual(attempts, list(RUNTIMES[:selected + 1]))
                factory.assert_called_once_with(model_path='/cached/model.tflite', num_threads=3)
                interpreter.allocate_tensors.assert_called_once()
                download.assert_not_called()

    def test_missing_runtimes_reports_desktop_install_command(self):
        original_import = builtins.__import__
        attempts = []

        def import_runtime(name, *args, **kwargs):
            if name in RUNTIMES:
                attempts.append(name)
                raise ImportError(name)
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=import_runtime), \
                patch.object(PoseDetector, '_get_model') as get_model, \
                self.assertLogs('pistream.detect', level='ERROR') as logs:
            detector = PoseDetector()
        self.assertFalse(detector.ready)
        get_model.assert_not_called()
        self.assertEqual(attempts, list(RUNTIMES))
        self.assertIn('pip install -e ".[desktop]"', '\n'.join(logs.output))
