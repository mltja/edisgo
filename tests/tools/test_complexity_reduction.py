import logging
import sys
import os
import pytest

from edisgo.tools.complexity_reduction import *
from edisgo import EDisGo


class TestComplexityReduction:

    @classmethod
    def setup_class(cls):
        cls.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_path,
            worst_case_analysis="worst-case"
        )

    def test_transform_coordinates(self):
        coor_berlin = (52.518611, 13.408333)
        coor_berlin_transformed = coor_transform.transform(coor_berlin[0], coor_berlin[1])
        coor_berlin_transformed_back = coor_transform_back.transform(coor_berlin_transformed[0], coor_berlin_transformed[1])
        assert round(coor_berlin[0],5) == round(coor_berlin_transformed_back[0],5)
        assert round(coor_berlin[1],5) == round(coor_berlin_transformed_back[1],5)

    def test_test(self):
        self.edisgo.analyze()
        print(self.edisgo.results.pfa_p)


def test_setup_logger(capsys):
    def check_stream_output(output):
        captured = capsys.readouterr()
        assert captured.out == output

    def check_file_output(output):
        with open('edisgo.log', 'r') as file:
            last_line = file.readlines()[-1].split(' ')[3:]
            last_line = ' '.join(last_line)
        assert last_line == output

    def reset_loggers():
        logger = logging.getLogger('edisgo')
        logger.propagate = True
        logger.handlers.clear()
        logger = logging.getLogger()
        logger.handlers.clear()

    setup_logger(stream_level=logging.DEBUG, file_level=logging.DEBUG, filename='edisgo.log', root=False)
    with capsys.disabled():
        print('\nLogger = edisgo')
    logger = logging.getLogger('edisgo')

    logger.debug("edisgo")
    check_stream_output("edisgo - DEBUG: edisgo\n")
    check_file_output("edisgo - DEBUG: edisgo\n")

    logging.debug("edisgo")
    check_stream_output("")
    check_file_output("edisgo - DEBUG: edisgo\n")

    reset_loggers()

    setup_logger(stream_level=None, file_level=logging.DEBUG, filename='edisgo.log', root=False)
    with capsys.disabled():
        print('Logger = edisgo - nostream')

    logger = logging.getLogger('edisgo')

    logger.debug("edisgo - nostream")
    check_stream_output("")
    check_file_output("edisgo - DEBUG: edisgo - nostream\n")

    logging.debug("edisgo - nostream")
    check_stream_output("")
    check_file_output("edisgo - DEBUG: edisgo - nostream\n")

    reset_loggers()

    setup_logger(stream_level=logging.DEBUG, file_level=logging.DEBUG, filename='edisgo.log', root=True)
    with capsys.disabled():
        print('Logger = root')
    logger = logging.getLogger('edisgo')

    logger.debug("root")
    check_stream_output("edisgo - DEBUG: root\n")
    check_file_output("edisgo - DEBUG: root\n")

    logging.debug("root")
    check_stream_output("root - DEBUG: root\n")
    check_file_output("root - DEBUG: root\n")

    os.remove("edisgo.log")




