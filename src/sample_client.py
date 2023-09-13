import aiohttp
import asyncio
import cv2
import base64
import argparse
from pathlib import Path
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, required=True, help="url of the web app")
    args = parser.parse_args()
    return args


async def send_post_request(url):
    test_imgname = 'dog.jpg'
    project_path = Path(__file__).parent.parent
    test_imgname = os.path.join(str(project_path), 'test_img', test_imgname)
    img = cv2.imread(test_imgname)
    _, encoded_image = cv2.imencode('.jpg', img)
    encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
    task = {'img': encoded_image_base64}

    async with aiohttp.ClientSession() as session:
        for i in range(1000):
            async with session.post(url, json=task) as response:
                if response.status == 200:
                    response_data = await response.json()
                    print("Response:", response_data)
                else:
                    print("Error:", response.status)


args = get_args()
loop = asyncio.get_event_loop()
loop.run_until_complete(send_post_request(args.url))
