# vLLM on AWS: build cheap, serve on GPU

Full walkthrough (VPC, SSH, disk, gotchas, validation): [RUNBOOK.md](./RUNBOOK.md)

Two scripts: [setup-vllm.sh](./setup-vllm.sh) installs vLLM on a cheap x86 CPU box so you can save an AMI, and [serve-vllm.sh](./serve-vllm.sh) starts vLLM on a g5 launched from that AMI. Default install is a prebuilt wheel (`pip install vllm==0.34.0`) with `TMPDIR=~/piptmp` on disk. Use `BUILD_FROM_SOURCE=1 bash setup-vllm.sh` for an editable checkout with precompiled kernels.

SSH user is `ec2-user` on Amazon Linux and `ubuntu` on Ubuntu AMIs. The commands below use `ec2-user` (this run).

| Step | Where | Command |
|---|---|---|
| 1. Copy scripts up | Your Mac | `scp -i ~/.ssh/inference-first.pem setup-vllm.sh serve-vllm.sh ec2-user@<build-ip>:~` |
| 2. Install | Build box (t3a.large, x86 DLAMI) | `bash ~/setup-vllm.sh` |
| 3. Save image | EC2 console | Instances > Actions > Image and templates > Create image, then terminate the build box |
| 4. Launch GPU | EC2 console | g5.xlarge or g5.2xlarge from your AMI, same subnet and security group |
| 5. Serve | g5 | `~/bin/serve-vllm.sh` (this run: `MAX_LEN=4096 ./serve-vllm.sh`; override model with `MODEL=Qwen/Qwen2.5-3B-Instruct`) |
| 6. Tunnel + test | Your Mac | `ssh -i ~/.ssh/inference-first.pem -L 8000:localhost:8000 ec2-user@<g5-ip>` then `curl localhost:8000/v1/models` |

## Test a completion

```
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Explain PagedAttention in one sentence."}]}'
```

## Notes
- The build box must be x86 (t3/t3a/c5), not Graviton, or the AMI won't boot on a g5.
- Use the Deep Learning Base OSS Nvidia Driver GPU AMI as the base: NVIDIA drivers and CUDA are preinstalled, so the g5 just works.
- vLLM binds to `127.0.0.1`; only port 22 is open in the security group and you reach 8000 through the SSH tunnel.
- Stop or terminate the g5 when you're done; deregister old AMIs and delete their snapshots.

## Chat in the browser

Keep the SSH tunnel open on your Mac, then start a tiny local server that serves the chat page and proxies `/v1` to vLLM. That keeps the request same-origin, so you don't hit CORS.

```
ssh -i ~/.ssh/inference-first.pem -L 8000:localhost:8000 ec2-user@<g5-ip>
```

In another terminal, from this folder:

```
python3 chat-ui/chat-server.py
```

Open http://localhost:3000. Enter sends, Shift+Enter adds a new line. The page loads the model name from `/v1/models`, streams the reply, and shows elapsed time plus token count and tokens/sec when vLLM sends a final usage chunk.

The server binds to localhost only. Override the ports if you need to: `PORT=3000 VLLM_URL=http://localhost:8000 python3 chat-ui/chat-server.py`.

With the tunnel up you can also open http://localhost:8000/docs for vLLM's Swagger page.
