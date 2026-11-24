# PROGRESS

โปรเจกต์: แปลงคลิปเสียงประชุม → รายงานการประชุมภาษาไทย
เริ่ม: 2026-08-07

## สภาพเครื่อง (มีผลกับการตัดสินใจทางเทคนิค)

- GPU: RTX 5060 Laptop, 8 GB VRAM, **Blackwell / sm_120** → บังคับใช้ torch cu128
- driver 580.173.02, ไม่มี `nvcc` (ไม่จำเป็น เพราะ torch wheel มี CUDA runtime มาให้)
- Python ระบบเป็น 3.14 → โปรเจกต์ pin 3.12 ใน venv แยก (ecosystem ML ยังตามไม่ทัน 3.14)
- `uv` เป็น tooling มาตรฐานของโปรเจกต์นี้

## Milestones

- [x] สำรวจเครื่อง เลือก stack (transformers + Typhoon Whisper + pyannote + Agent SDK)
- [x] ยืนยัน Claude Agent SDK API จาก source ของ package จริง (v0.2.132)
      → `query(prompt=..., options=ClaudeAgentOptions(...))`, `tools=[]`,
        `setting_sources=[]` สำหรับ isolation
- [x] เขียน pipeline ครบ 5 ขั้น: audio / transcribe / diarize / merge / summarize
- [x] เขียน prompt เทมเพลตรายงานการประชุมภาษาไทย
- [x] เขียน CLI + README
- [x] ติดตั้ง dependencies (`uv sync`) — torch 2.11.0+cu128, `torch.cuda.is_available()` = True,
      การ์ดขึ้นเป็น sm_120 ถูกต้อง
- [x] ติดตั้ง ffmpeg — `/usr/bin/ffmpeg` เวอร์ชัน 8.0.1
- [x] แก้ pyannote พังตอน import (torchaudio 2.11 ถอด API ที่ 3.x ใช้) → ย้ายไป 4.x
- [x] เลี่ยง torchcodec ที่ build มากับ CUDA 13 → ป้อน waveform ตรงเข้า pipeline
- [x] เขียนชุดเทสต์ offline 47 เคส (`uv run pytest`) ครอบคลุม merge / timestamp /
      transcript round-trip / prompt assembly
- [x] ต้องยก torch/torchaudio/torchcodec ทั้ง stack ไป PyPI default (CUDA 13) —
      cu128 index เดิมหยุดออก torchcodec หลัง 0.11.1 ทำให้ CUDA เพี้ยนกับ torch
      → transformers พัง import torchcodec ตอนถอดเสียงจริง (ไม่ใช่แค่ตอน diarize)
- [x] เจอบั๊กร้ายแรง: 44 นาทีถูกยุบเหลือ chunk เดียว เพราะ pipeline's
      `chunk_length_s` (long-form auto-split) ไม่น่าเชื่อถือกับ checkpoint นี้ →
      เปลี่ยนไปหั่น window 30 วิเอง ดูรายละเอียดในบันทึกด้านล่าง
- [x] ทดสอบกับไฟล์เสียงจริง (การนำเสนอโปรเจกต์ 43.8 นาที) — วัดเวลา
      ถอดเสียงจริงแล้ว (ดูตัวเลขในบันทึก), timestamp ราย 30 วิถูกต้อง
- [ ] ตั้งค่า HF_TOKEN + ยอมรับ license pyannote (ถ้าต้องการแยกผู้พูด) — **ต้องให้ผู้ใช้ทำเอง**
- [ ] ปรับ prompt จากผลลัพธ์จริง (รอบแรกออกมาดีมาก ดูบันทึกด้านล่าง)

## บันทึกการตัดสินใจ

**ทำไมไม่ใช้ faster-whisper**
ตอนแรกตั้งใจใช้ แต่ RTX 5060 เป็น sm_120 ซึ่ง CTranslate2 (backend ของ
faster-whisper) ซัพพอร์ตยังไม่นิ่ง และต้องแปลง Typhoon checkpoint เป็นฟอร์แมต
CT2 เพิ่มอีกขั้น → เลือก transformers ซึ่งรันเช็คพอยต์ที่เผยแพร่ได้ตรง ๆ
แลกกับช้ากว่าราว 30-40%

**ทำไมเก็บ transcript.json แยก**
ถอดเสียงคือขั้นที่แพงที่สุด ส่วนที่อยากทำซ้ำบ่อยคือปรับ prompt
การเก็บ intermediate ทำให้ `--from-transcript` วนรอบ prompt ได้ในไม่กี่วินาที

**ทำไมถึงต้องขยับ pyannote ขึ้น 4.x ทั้งที่เดิมตั้งใจ pin `<4`**
pin เดิมเขียนไว้ว่า "เพราะ pipeline 3.1 ใช้ API ของ 3.x" แต่พอลงจริงกลับ import
ไม่ผ่านเลย: pyannote 3.4 อ้าง `torchaudio.AudioMetaData` กับ `torchaudio.info`
ตั้งแต่ระดับ module ส่วน torchaudio 2.9 ขึ้นไปถอดทั้งคู่ออกแล้ว
ทางเลือกมีสองทาง — ถอย torch ลงให้ต่ำกว่า cu128 (ซึ่งแปลว่าการ์ดใช้ไม่ได้) หรือ
ขยับ pyannote ขึ้น จึงเลือกอย่างหลัง ข้อสังเกตคือ pin ที่ตั้งใจให้ปลอดภัย
กลายเป็นตัวปัญหาเอง เพราะมันถูกเขียนจากสมมติฐาน ไม่ได้เขียนจากการลองลงจริง

**ทำไมส่ง waveform ให้ pyannote แทน path ของไฟล์**
pyannote 4.x decode ไฟล์ด้วย torchcodec ซึ่ง wheel บน PyPI ลิงก์กับ CUDA 13
(`libcudart.so.13`) คนละตัวกับ torch cu128 ของเรา — และ index cu128 ก็มี
torchcodec แค่ถึง 0.11.1 การไปไล่จับคู่เวอร์ชัน torchcodec↔torch↔FFmpeg
คือหนี้ที่จะกลับมาพังทุกครั้งที่อัปอะไรสักอย่าง
แต่ pyannote เปิดทางไว้ให้ป้อน `{"waveform", "sample_rate"}` ตรง ๆ ซึ่งข้ามชั้น
decode ทั้งชั้น (ตรวจแล้วว่า torchcodec ถูกอ้างเฉพาะใน `core/io.py` และอยู่หลัง
branch ของ path ล้วน) ผลพลอยได้สำคัญกว่านั้นคือ ffmpeg กลายเป็นตัว decode
ตัวเดียวของทั้งระบบ — ข้อสมมติเรื่อง "timeline เดียวกัน" ที่ `merge.py`
ตั้งอยู่บนนั้น จึงเป็นจริงโดยโครงสร้าง ไม่ใช่เพราะบังเอิญ resample ตรงกัน

**ทำไมถอด cu128 index ออกทั้งบล็อก แทนที่จะขยับไป cu130 แบบ pin เอง**
เช็ค metadata ของ torch 2.11.0 บน PyPI พบว่ามันดึง `nvidia-*-cu13` เป็น
dependency อยู่แล้ว — PyPI เปลี่ยน default เป็น CUDA 13 ไปเอง ไม่ต้องมี custom
index อีกต่อไป การ pin เองจะเป็นภาระบำรุงรักษาที่ไม่จำเป็น (ต้องคอยเช็คว่า
cu130 index ยังมี torchcodec เวอร์ชันที่ตรงกับ torch ไหมทุกครั้งที่อัป)
ในขณะที่ปล่อยให้ทุก wheel มาจาก PyPI เดียวกัน ทำให้ CUDA version ตรงกันโดย
โครงสร้าง ข้อสังเกตสำคัญ: บล็อกที่เขียนไว้เพื่อ "กันปัญหา" (cu128 pin) พอเวลา
ผ่านไปกลับกลายเป็นตัวปัญหาเอง เพราะ ecosystem เปลี่ยนเร็วกว่าที่คอมเมนต์ตอน
เขียนจะทัน — ต้องเช็คของจริงทุกครั้ง ไม่ใช่เชื่อคอมเมนต์เก่า

**บั๊กร้ายแรง: 44 นาทีถูกถอดเสียงออกมาเป็น chunk เดียว**
รอบแรกที่รันไฟล์จริง (การนำเสนอโปรเจกต์ 43.8 นาที) transcript.json
ออกมามี chunk เดียว ครอบคลุม 0-2622 วินาที ข้อความยาว 32,279 ตัวอักษรในก้อน
เดียว — timestamp ในรายงานเลยกลายเป็น [00:00:00] หมดทุกจุด และถ้าเปิด
diarization จะยิ่งพังหนักกว่านั้น เพราะ `merge.py` จับคู่ผู้พูดต่อ chunk
ไม่ใช่ต่อคำ — chunk เดียวครอบทั้งไฟล์แปลว่าทุกคนในห้องจะถูกยุบเหลือผู้พูด
คนเดียว

ต้นตอ: เดิม `transcribe.py` ส่งเสียงทั้งไฟล์เข้า `pipeline(chunk_length_s=30)`
ให้ transformers หั่นเองและต่อกลับ (long-form auto-split) แต่ transformers
เตือนเองตรง ๆ ว่ากลไกนี้ "very experimental" สำหรับ seq2seq model และแนะนำ
ให้เรียก `generate()` เองแทน ทดสอบแล้วพบว่า checkpoint ตัวนี้ (Typhoon
Whisper) ไม่เคยทำนาย closing timestamp token ได้เลยตลอดทั้งไฟล์ — ตัว
boundary-detection ของ transformers (หา timestamp token คู่ติดกัน) เลยไม่เคย
ทำงาน ผลคือทุกอย่างถูกต่อเป็นก้อนเดียว

วิธีแก้: หั่นเสียงเป็น window 30 วินาทีเอง (`_split_into_windows`) แล้วใช้
**ขอบเขตของ window ที่เรารู้อยู่แล้วเป็น timestamp** แทนที่จะรอให้โมเดล
ทำนายเอง — 30 วินาทีเลือกเพราะ encoder ของ Whisper ประมวลผลเป็นเฟรม mel
คงที่ 30 วิเสมอ (เสียงสั้นกว่าจะถูก pad ด้วยความเงียบ) การหั่นเล็กกว่านี้จึง
ไม่ได้ช่วยอะไรนอกจาก timestamp ละเอียดขึ้น แลกกับเวลารันเพิ่มเป็นสัดส่วน
(30/window_size) เพราะ encoder cost เท่าเดิมทุกครั้งไม่ว่าจะป้อนเสียงสั้นแค่
ไหน — ผลข้างเคียงที่ดีคือ warning "chunk_length_s is very experimental" ก็
หายไปด้วย เพราะไม่เรียกกลไกนั้นอีกแล้ว

**word-level timestamp ใช้ไม่ได้บนการ์ดนี้เลย ไม่ว่าจะตั้ง batch-size เท่าไหร่**
ทดสอบแยกแล้ว: แม้แยก audio มาแค่ window เดียว 30 วิ (ไม่มี accumulation จาก
ไฟล์ยาว) word-level timestamp ก็ OOM ทันทีบนการ์ด 8 GB — cross-attention
alignment matrix ที่ต้องเก็บไว้คำนวณ word alignment ใหญ่เกินไปสำหรับ card
รุ่นนี้ และเป็นข้อจำกัดที่**ตายตัวตาม hardware** ไม่ขึ้นกับเนื้อหาเสียงหรือ
batch-size เลย (`--batch-size` แทบไม่ขยับตัวเลข VRAM ที่ใช้ตอน OOM เลยตอน
ทดสอบ) เพราะงั้นคำแนะนำเดิมใน README ("ลด --batch-size ถ้า OOM") ใช้ไม่ได้
กับปัญหานี้ — แก้แล้วโดยให้ระบบลองครั้งเดียวแล้วปิด word-level ถาวรสำหรับ
ไฟล์ที่เหลือ แทนที่จะลองใหม่ทุก window (ของเดิมจะลองซ้ำ ๆ ที่ตายซ้ำ ๆ
เสียเวลาฟรี)

**คุณภาพผลลัพธ์รอบแรกกับไฟล์จริง**
ทดสอบกับไฟล์นำเสนอโปรเจกต์จริง (ภาษาไทยผสมศัพท์เทคนิคอังกฤษ เช่น
PostgreSQL, MongoDB, Prisma ORM) — สรุปออกมาแม่นมาก จัดกลุ่มตามหัวข้อถูกต้อง
ระบุ "ไม่มีมติ" ตรงไปตรงมาเพราะเนื้อหาเป็น Q&A ล้วน และที่สำคัญ Claude
เตือนปัญหา timestamp เองในหมายเหตุท้ายเอกสารตอนที่ transcript ยังมีปัญหาอยู่
(รอบก่อนแก้บั๊ก) แทนที่จะมั่ว timestamp ปลอมมาให้ — กฎ "ทุกมติต้องมี
timestamp อ้างอิงกลับไปฟังได้" ใน prompt ทำงานตามที่ตั้งใจไว้จริง

**ทำไม summarize ใช้ `tools=[]` และ `setting_sources=[]`**
เป็น text transformation ล้วน ไม่ต้องการ tool ใด ๆ และไม่ควรให้ CLAUDE.md
ของ directory ที่บังเอิญรันอยู่มาเปลี่ยนสไตล์ผลลัพธ์เงียบ ๆ
→ ทำให้ output เป็นฟังก์ชันของ (prompt, transcript) เท่านั้น
