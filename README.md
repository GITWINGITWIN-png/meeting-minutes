# meeting-minutes

**English summary:** turns Thai meeting audio into written minutes. Speech recognition (Typhoon Whisper, a Thai fine-tune of Whisper) and speaker diarization (pyannote) run locally on a laptop GPU, then an LLM writes the minutes from the merged transcript, citing a timestamp for every decision. Long recordings are cut into 30-second windows because the model's own long-form splitting collapsed a 44-minute file into one chunk. The rest of this README is in Thai.

แปลงคลิปเสียงการประชุมให้เป็นรายงานการประชุมภาษาไทย ทำงานบนเครื่องตัวเอง (GPU)
ยกเว้นขั้นตอนสรุปที่เรียก Claude ผ่าน Claude Code login เดิม

```
คลิปเสียง → [ffmpeg] 16kHz mono → [Typhoon Whisper] transcript
                                → [pyannote] ใครพูดเมื่อไหร่
                                → merge → [Claude] รายงานการประชุม
```

## ทำไมเลือกแบบนี้

| ตัวเลือก | เหตุผล |
|---|---|
| `transformers` ไม่ใช่ `faster-whisper` | RTX 50-series คือ Blackwell (sm_120) — CTranslate2 ที่ faster-whisper ใช้ยังซัพพอร์ตไม่นิ่ง และต้องแปลงโมเดลเป็นฟอร์แมต CT2 ก่อน |
| torch จาก index `cu128` | wheel มาตรฐานบน PyPI build กับ CUDA เก่ากว่า จะ error `no kernel image is available` บน sm_120 |
| Typhoon Whisper ไม่ใช่ Whisper ดิบ | สถาปัตยกรรมเดียวกัน แต่ fine-tune ภาษาไทย — แม่นกว่ามากโดยไม่ต้องแก้โค้ด |
| ถอดเสียงก่อน แล้วค่อยสรุป | Claude ไม่รับ input เสียงโดยตรง และการแยกขั้นทำให้ปรับ prompt ซ้ำได้โดยไม่ต้องถอดเสียงใหม่ |
| pyannote 4.x ไม่ใช่ 3.x | 3.x เรียก `torchaudio.AudioMetaData` กับ `torchaudio.info` ตอน import แต่ torchaudio ถอดทั้งคู่ออกไปตั้งแต่ 2.9 — จะอยู่กับ 3.x ได้ต้องถอย torch ลงต่ำกว่า build cu128 ที่การ์ดใบนี้ต้องใช้ |
| ป้อน waveform ให้ pyannote ตรง ๆ ไม่ส่ง path | pyannote 4.x อ่านไฟล์ด้วย torchcodec ซึ่ง wheel บน PyPI ลิงก์กับ CUDA 13 คนละตัวกับ torch cu128 ของเรา — ป้อน array ที่ ffmpeg decode ไว้แล้วจะข้ามชั้นนั้นทั้งชั้น และได้ผลพลอยได้คือ decode ครั้งเดียวจริง ๆ ทำให้ timeline ตรงกันโดยโครงสร้าง |

## ติดตั้ง

```bash
sudo apt install ffmpeg      # ต้องรันเอง (ต้องใช้รหัสผ่าน)
uv sync                      # ลง torch cu128 + โมเดลอื่น ๆ
```

**ถ้าต้องการแยกผู้พูด** ต้องมี Hugging Face token เพราะโมเดล pyannote เป็น gated:

1. กดยอมรับ license ที่ https://hf.co/pyannote/speaker-diarization-3.1
2. กดยอมรับ license ที่ https://hf.co/pyannote/segmentation-3.0
3. สร้าง read token ที่ https://hf.co/settings/tokens
4. `export HF_TOKEN=hf_xxxxx`

ถ้าไม่ทำ pipeline จะยังทำงานได้ แค่ข้ามขั้นแยกผู้พูด (transcript ไม่มีป้ายชื่อ)

## ใช้งาน

```bash
# ครั้งแรก — ถอดเสียง แยกผู้พูด แล้วสรุป
uv run meeting-minutes ประชุม.m4a --speakers 4 --title "Sprint planning" --date 2026-08-07

# ปรับ prompt แล้วสรุปใหม่ (ไม่ต้องถอดเสียงซ้ำ — เร็วเป็นวินาที)
uv run meeting-minutes --from-transcript out/transcript.json

# เอาแค่ transcript ไม่ต้องสรุป
uv run meeting-minutes ประชุม.m4a --transcribe-only

# VRAM ไม่พอ
uv run meeting-minutes ประชุม.m4a --batch-size 2

# ไม่มี HF token
uv run meeting-minutes ประชุม.m4a --no-diarize
```

ผลลัพธ์ทั้งหมดอยู่ใน `out/`:

| ไฟล์ | คืออะไร |
|---|---|
| `audio_16k.wav` | เสียงที่ decode แล้ว (ทั้ง ASR และ pyannote ใช้ไฟล์เดียวกัน → timeline ตรงกัน) |
| `transcript.json` | ข้อมูลดิบ มี timestamp + speaker ต่อ chunk — ใช้กับ `--from-transcript` |
| `transcript.txt` | transcript อ่านง่าย `[HH:MM:SS] SPEAKER_00: ...` |
| `minutes.md` | รายงานการประชุม |

## ปรับคุณภาพรายงาน

คุณภาพของ `minutes.md` มาจาก `prompts/minutes_th.md` เป็นหลัก ไม่ใช่จาก ASR
แก้ไฟล์นั้นแล้วรัน `--from-transcript` ซ้ำเพื่อดูผลทันที

กฎสำคัญที่ใส่ไว้ในเทมเพลตแล้ว:
- จัดกลุ่มตามหัวข้อ ไม่ใช่ตามลำดับเวลา (คนประชุมพูดวกวน)
- แยก "มติ" ออกจาก "สิ่งที่คุยกัน" เด็ดขาด
- ห้ามเดาผู้รับผิดชอบหรือกำหนดเวลา — ให้เขียน `[ยังไม่ระบุ]` แทน
- ทุกมติต้องมี timestamp อ้างอิงกลับไปฟังได้

## เทสต์

```bash
uv run pytest
```

เทสต์ทั้งชุดไม่แตะ GPU ไม่แตะไฟล์เสียง ไม่ต่อเน็ต — รันจบในไม่กี่วินาที
ครอบคลุมเฉพาะส่วนที่ไม่มีโมเดลอยู่ข้างใน ได้แก่ การจับคู่ ASR กับผู้พูด
(`merge.py`) การซ่อม timestamp ที่ Whisper ให้มาไม่ครบ (`transcribe.py`)
การอ่าน-เขียน `transcript.json` และการประกอบ prompt

ประโยชน์จริงคือตอนรายงานออกมาเพี้ยน: ถ้าเทสต์ยังเขียว แปลว่าปัญหาอยู่ที่โมเดล
หรือที่ prompt ไม่ใช่ที่โค้ดเชื่อมต่อ

## ข้อจำกัดที่ควรรู้

- **ชื่อคน** pyannote บอกได้แค่ว่า "คนละคน" ไม่ได้บอกว่าเป็นใคร จึงออกมาเป็น
  `SPEAKER_00` ถ้าในบทสนทนามีการเรียกชื่อกัน Claude จะพยายามจับคู่ให้
- **จำนวนผู้พูด** ถ้ารู้ว่ามีกี่คนให้ใส่ `--speakers N` เสมอ การเดาผิดคือสาเหตุ
  อันดับหนึ่งที่ป้ายผู้พูดมั่ว
- **VRAM 8 GB** พอสำหรับตัวโมเดลเอง (fp16 ~2.9 GB) และพอสำหรับถอดเสียงแบบ
  chunk-level แต่ **ไม่พอสำหรับ word-level timestamp** — cross-attention
  alignment matrix ใหญ่เกินไปเสมอบนการ์ดขนาดนี้ ทดสอบแล้วว่าไม่ขึ้นกับ
  `--batch-size` เลย (ลดเท่าไหร่ก็ยัง OOM) ระบบจะลอง word-level ครั้งเดียว
  แล้วสลับไป chunk-level (ความละเอียด 30 วิ) ให้เองอัตโนมัติ ไม่ต้องตั้งค่า
  อะไรเพิ่ม — ถ้าเจอ OOM ที่ตัวโมเดลเองไม่ขึ้นเลย ให้สลับไป
  `--model typhoon-ai/typhoon-whisper-turbo` แทน

---

พัฒนาโดยมี AI coding assistants ช่วย (built with help from AI coding assistants)
