from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
import traceback
import uuid
import os
from logger import logger
from boundingRect import detect
import cv2

app = Flask(__name__)


@app.route('/identifyDirection', methods=['POST'])
def get_direction():
    """
    图片方向识别
    :return:
    """
    up_file_name = ''
    try:
        data_file = request.files['file']
        if data_file:
            up_file_name = data_file.filename
            file_name = f'{str(uuid.uuid4())}.{up_file_name.rsplit(".", 1)[1]}'
            if not os.path.exists('taskData'):
                os.makedirs('taskData')
            data_file.save(f'taskData/{file_name}')
            img = cv2.imread(f'taskData/{file_name}')
            direction, angle = detect(img, False)
            os.remove(f'taskData/{file_name}')
            return jsonify({'code': 0, 'msg': f'{up_file_name} 识别完成', 'data': {'direction': direction, 'angle': angle}})
        else:
            return jsonify({'code': 10001, 'msg': '未上传图片', 'data': ''})
    except Exception as e:
        logger.error(e)
        return jsonify({'code': 10001, 'msg': f'{up_file_name} 上传失败', 'data': ''})


def run(ser_port=1666):
    """
    启动服务
    :param ser_port:
    :return:
    """
    api_server = None
    try:
        api_server = WSGIServer(('0.0.0.0', ser_port), app)
        api_server.log.write(f'服务启动成功 http://127.0.0.1:{ser_port}/')
        api_server.serve_forever()
    except Exception as e:
        traceback.print_stack()
    finally:
        if api_server:
            api_server.close()


if __name__ == '__main__':
    app.json_encoder.ensure_ascii = False
    run(38082)
