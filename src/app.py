import asyncio
from aiohttp import web
from task_manager import TaskScheduler
import argparse
from util import parse_config
from handle import registry

handle_list = registry.class_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='config file of app')
    parser.add_argument('--port', '-p', type=int, default=9876)
    args = parser.parse_args()
    return args


args = get_args()
config_file = args.config
config = parse_config(config_file)
gpu_id = config['GPUs'].split(',')
_handle = handle_list[config['Handle']]
ts = TaskScheduler(config['ProcessPerGPU'], gpu_id, _handle)


async def predict(request):
    try:
        data = await request.json()  # parse the json data
    except Exception as e:
        error_response = {"error": "Failed to process JSON data"}
        return web.json_response(error_response, status=400)
    task = data
    # put to task queue
    worker_id, task_id = ts.add_task(task)
    # wait until finished
    while True:
        state = ts.check_state(worker_id, task_id)
        if state:
            result = ts.get_result(worker_id, task_id)
            break
        await asyncio.sleep(0.02)
    return web.json_response(result)


app = web.Application()
app.add_routes([web.post('/predict', predict)])

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=args.port)
