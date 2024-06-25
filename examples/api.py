''' 生成api server的示例代码
'''
from bert4vector.pipelines import EmbeddingSever, EmbeddingClientRequest, EmbeddingClientAiohttp


def start_server(port):
    # server端
    server = EmbeddingSever('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
    server.run(port=port)

def start_client_request(port):
    # client端
    client = EmbeddingClientRequest(base_url=f'http://0.0.0.0:{port}')
    print(client.add_corpus(['你好', '天气不错']))
    print(client.encode('你好'))
    print(client.similarity(query1='你好', query2='你好啊'))
    print(client.search('你好'))
    print(client.summary())

async def start_client_aiohttp(port):
    # client端
    client = EmbeddingClientAiohttp(base_url=f'http://0.0.0.0:{port}')
    print(await client.add_corpus(['你好', '天气不错']))
    print(await client.encode('你好'))
    print(await client.similarity(query1='你好', query2='你好啊'))
    print(await client.search('你好'))
    print(await client.summary())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8001)
    parser.add_argument('--mode', default='client_aiohttp', choices=['server', 'client_request', 'client_aiohttp'])
    args = parser.parse_args()
    port = int(args.port)

    if args.mode == 'server':
        start_server(port)
    elif args.mode == 'client_request':
        start_client_request(port)
    elif args.mode == 'client_aiohttp':
        import asyncio
        asyncio.run(start_client_aiohttp(port))