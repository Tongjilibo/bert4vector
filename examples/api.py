''' 生成api server的示例代码
'''
from bert4vector.pipelines import SimilaritySever, SimilarityClientRequest, SimilarityClientAiohttp


def start_server(port):
    # server端
    server = SimilaritySever('/data/pretrain_ckpt/BAAI/bge-base-zh-v1.5')
    server.run(port=port)

def start_client_request(port):
    # client端
    client = SimilarityClientRequest(base_url=f'http://0.0.0.0:{port}')
    print(client.add_corpus(['你好', '天气不错']))
    print(client.encode('你好'))
    print(client.similarity(query1='你好', query2='你好啊'))
    print(client.search('你好'))
    print(client.delete_corpus(['天气不错']))
    print(client.summary())
    print(client.reset())
    print(client.summary())

async def start_client_aiohttp(port):
    # client端
    client = SimilarityClientAiohttp(base_url=f'http://0.0.0.0:{port}')
    print(await client.add_corpus(['你好', '天气不错']))
    print(await client.encode('你好'))
    print(await client.similarity(query1='你好', query2='你好啊'))
    print(await client.search('你好'))
    print(await client.delete_corpus(['天气不错']))
    print(await client.summary())
    print(await client.reset())
    print(await client.summary())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=9000)
    parser.add_argument('--mode', default='server', choices=['server', 'client_request', 'client_aiohttp'])
    args = parser.parse_args()
    port = int(args.port)

    if args.mode == 'server':
        start_server(port)
    elif args.mode == 'client_request':
        start_client_request(port)
    elif args.mode == 'client_aiohttp':
        import asyncio
        asyncio.run(start_client_aiohttp(port))