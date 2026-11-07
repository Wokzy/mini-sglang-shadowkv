"""Exercise a running Mini-SGLang server; no additional client dependencies.

Launch with README's ArkVale configuration first. Use small budgets to ensure
long prompts exceed residency. Server release logs must show nonzero evictions
and score_steps; API success alone does not prove the recall path executed.
"""

import argparse
import concurrent.futures
import json
import time
import urllib.request


def request(base_url, model, prompt, max_tokens=80, ignore_eos=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
    }
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        assert response.status == 200
        result = json.load(response)
    choice = result["choices"][0]
    content = choice["message"]["content"]
    assert isinstance(content, str) and content.strip(), result
    assert choice["finish_reason"] is not None, result
    print(json.dumps({"output": content[:200], "finish_reason": choice["finish_reason"]}))
    return result


def cancel_stream(base_url, model, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "ignore_eos": True,
        "stream": True,
    }
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        assert response.status == 200
        assert response.readline().startswith(b"data: ")
    # Closing a streaming response triggers the API's AbortMsg path.
    time.sleep(2)
    print("Disconnected an active streaming request")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    request(args.base_url, model, "What is the capital of France? Answer briefly. /no_think")
    long_prompt = (
        "The archive contains notes about rivers and mountains. " * args.repetitions
        + "Summarize the topic of these notes in one sentence. /no_think"
    )
    # Force enough decode steps to cross page boundaries, even after EOS.
    request(args.base_url, model, long_prompt, ignore_eos=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(request, args.base_url, model, prompt, ignore_eos=ignore_eos)
            for prompt, ignore_eos in (
                (long_prompt, True),
                ("Name three primary colors. Answer briefly. /no_think", False),
            )
        ]
        for future in futures:
            future.result()
    cancel_stream(args.base_url, model, long_prompt)
    # Completed/aborted requests' scheduler slots should now be reused cleanly.
    request(args.base_url, model, "What is 2 + 2? Answer briefly. /no_think")
    print(
        "PASS: five requests completed with nonempty outputs; streaming cancellation and reuse exercised"
    )


if __name__ == "__main__":
    main()
