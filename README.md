# ต้นแบบตรวจจับตำหนิ PCB ด้วย YOLO11

แอปเดสก์ท็อปสำหรับเลือกภาพ PCB ทีละภาพ แสดงกรอบตรวจจับและรายละเอียด แจ้งเตือนเมื่อพบตำหนิ และเก็บภาพผลกับ SQLite เพื่อเรียกดูย้อนหลัง ใช้สำหรับเรียนรู้ตาม `report/third draft.pdf` ไม่กำหนดคะแนนความแม่นยำหรือความเร็วขั้นต่ำ

**สถานะวันที่ 15 ตุลาคม 2026:** พัฒนาแอปและโค้ดเตรียมฝึก/ประเมินแล้ว **ยังไม่ได้เทรนโมเดล** และยังไม่มีน้ำหนัก PCB สำหรับตรวจจริง ขณะนี้ทดสอบหน้าจอได้ด้วยโหมดจำลองซึ่งไม่บันทึกภาพหรือประวัติจริง

## เริ่มใช้งานบน Windows

เปิด PowerShell ที่โฟลเดอร์โครงการ ใช้ environment ที่มีอยู่:

```powershell
.\.venv\Scripts\python.exe main.py --mock
```

เลือก JPG/JPEG/PNG หนึ่งภาพ แล้วกด **ตรวจจับ** โหมดจำลองแสดงกล่องตัวอย่างสองประเภทเพื่อทดสอบภาพ ตาราง Dialog และการซูม สามารถทดลองปรับ Confidence เป็น `0.99` เพื่อทดสอบกรณีไม่พบกรอบ ผลเหล่านี้ไม่ใช่การทำนาย PCB จากโมเดลที่ฝึกแล้ว

เมื่อมีโมเดลพร้อม ให้เปิดโหมดจริงด้วย:

```powershell
.\.venv\Scripts\python.exe main.py
```

ค่าเริ่มต้นอ่าน `models/pcb/best.pt` และ `models/pcb/metadata.json` หากไฟล์ยังไม่มีหรือ mapping/checksum ผิด แอปแสดงว่าโมเดลไม่พร้อมและปิดปุ่มตรวจจับ สามารถเลือกชุดโมเดลอื่นผ่านปุ่ม **เลือกโมเดล / โหลดใหม่** หรือใช้ `--model <path-to-best.pt>` โดย metadata ต้องอยู่ในโฟลเดอร์เดียวกัน

## ตรวจภาพทีละภาพด้วย OpenVINO บน CPU

ใช้ OpenVINO FP32 สำหรับตรวจภาพบน Intel CPU โดยโมเดล `.pt` เดิมยังใช้ได้ ติดตั้งและแปลงชุดโมเดลที่มี `metadata.json` แล้ว เช่นโมเดล 2 epochs:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-openvino.txt
.\.venv\Scripts\python.exe -m scripts.export_openvino --weights models/pcb-2epochs-ui/best.pt --output models/pcb-2epochs-openvino
.\.venv\Scripts\python.exe main.py --model models/pcb-2epochs-openvino/best_openvino_model.xml
```

สคริปต์สร้างโฟลเดอร์ใหม่พร้อม `best_openvino_model.xml`, `best_openvino_model.bin`, `metadata.yaml` และ `metadata.json` ตรวจ checksum ของไฟล์ที่ใช้รันก่อนโหลด ไม่เขียนทับโฟลเดอร์ที่มีอยู่แล้ว การแปลงใช้ CPU และไม่ต้องเทรนใหม่ ใน UI เลือก `.xml` ได้ด้วยปุ่มเลือกโมเดล ต้องเก็บไฟล์ทั้งชุดไว้ด้วยกัน ชื่อ XML ต้องมี `_openvino_model` เพื่อให้ Ultralytics เวอร์ชันนี้รู้จักรูปแบบ

โมเดล OpenVINO ใช้ CPU โดยระบุ `intel:cpu` แม้มี NVIDIA GPU ใช้งานได้ มีการ compile/warm-up ตอนโหลดโมเดลใน worker ก่อนเปิดให้ตรวจภาพ รองรับขนาดภาพเข้าโมเดลที่ปรับได้ผ่าน dynamic shapes กรอบ คลาส confidence การบันทึกหลักฐานและประวัติใช้ขั้นตอนเดิม ความเร็วจริงขึ้นอยู่กับเครื่องและงานที่กำลังทำพร้อมกัน ควรเทียบผลตรวจและ validation กับ `.pt` ก่อนใช้จริง; FP32 ไม่รับประกันว่าผลตัวเลขจะตรงทุกหลัก

ทดสอบเวลา CPU และความสอดคล้องของกรอบบนตัวอย่าง validation 10 ภาพที่กระจายตามลำดับไฟล์ หลัง warm-up ภาพละ 3 ครั้ง:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_openvino --reference models/pcb-2epochs-ui/best.pt --openvino models/pcb-2epochs-openvino/best_openvino_model.xml
```

ผลอยู่ใน `reports/openvino_benchmark.json` รายงาน median เวลาอ่าน/ตรวจ/วาดภาพ (ไม่รวม UI หรือการบันทึกลงดิสก์), เวลา inference และจำนวนกรอบที่คลาสตรงกันและ IoU ≥ 0.99 การตรวจตัวอย่างนี้ไม่แทนการประเมิน mAP/Recall บน validation ทั้งชุด และเวลาขณะเทรนพร้อมกันอาจเปลี่ยนได้

## ติดตั้งบนเครื่องใหม่

### Clone เพื่อเทรนอีกเครื่อง

Repository เก็บโค้ด configuration คู่มือ ชุดทดสอบซอฟต์แวร์ พร้อม ZIP ชุดข้อมูล `PCB Defect.v1-yolov5.yolov11.zip` และน้ำหนักตั้งต้น `models/pretrained/yolo11n.pt` ผ่าน Git LFS ส่วนโฟลเดอร์ dataset ที่แตกแล้ว น้ำหนักที่เทรนเอง ผลการเทรน TensorBoard log virtual environment และรายงานต้นฉบับใน `report/` ไม่ถูกอัปโหลด

ติดตั้ง Git และ Git LFS บนเครื่องใหม่ก่อน แล้วรัน:

```powershell
git lfs install
git clone https://github.com/Jadsadar/Yolo11_PCB_pre-operator.git
cd Yolo11_PCB_pre-operator
git lfs pull
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

บน repo ที่ clone ใหม่ แตก dataset ไว้ในโฟลเดอร์ชั่วคราว แล้วนำเฉพาะโฟลเดอร์ `train/`, `valid/`, `test/` มาวางข้าง `data.yaml` ของ repo ให้แต่ละโฟลเดอร์มี `images/` และ `labels/` ตามผังด้านล่าง ใช้ `data.yaml` ที่มากับ repo เนื่องจากแก้ path ให้ตรงโครงสร้างแล้ว:

```powershell
Expand-Archive -LiteralPath '.\PCB Defect.v1-yolov5.yolov11.zip' -DestinationPath '.\tmp\dataset'
Move-Item -Path '.\tmp\dataset\train', '.\tmp\dataset\valid', '.\tmp\dataset\test' -Destination '.'
```

หาก ZIP มีขนาดเพียงร้อยกว่าไบต์และเปิดไม่ได้ แสดงว่ายังเป็น LFS pointer ให้รัน `git lfs pull` เพื่อดาวน์โหลดไฟล์ข้อมูลจริงก่อนแตก archive

```powershell
# ตรวจ checksum น้ำหนักที่รับผ่าน Git LFS (ดาวน์โหลดจากต้นทางหากไฟล์ยังไม่มี)
.\.venv\Scripts\python.exe -m scripts.download_pretrained

# ตรวจชุดข้อมูลและสร้างแผนฝึกของเครื่องนี้
.\.venv\Scripts\python.exe -m scripts.validate_dataset
.\.venv\Scripts\python.exe -m scripts.train
```

แก้ `configs/train.yaml` ให้เหมาะกับ CPU/GPU ของเครื่องใหม่ก่อนใช้ `--execute` หากต้องการ GPU ต้องติดตั้ง PyTorch/torchvision แบบ CUDA ให้ตรงกับเครื่อง ชุดทดสอบซอฟต์แวร์ใน `tests/` ใช้ภาพชั่วคราวของตัวเองและไม่ต้องมี dataset จริง

### Environment ของแอป

ใช้ Python 3.10 ขึ้นไป เวอร์ชันที่ตรวจในงานนี้คือ Python 3.10.8 บน Windows และ PyTorch แบบ CPU ถ้าเครื่องยังไม่มี virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

ไฟล์ requirements ตรึงเวอร์ชันที่ติดตั้งและผ่านการตรวจซอฟต์แวร์ใน workspace นี้ การติดตั้ง dependency ต้องใช้อินเทอร์เน็ต แต่แอปตรวจภาพทำงานออฟไลน์หลังเตรียมโมเดลแล้ว Runtime เก็บค่า Ultralytics ที่ `.runtime/` และปิดการดาวน์โหลด dependency อัตโนมัติ สำหรับ GPU ให้ติดตั้ง PyTorch/torchvision รุ่น CUDA ที่เข้าคู่กันจากคำแนะนำของ PyTorch และบันทึก environment ของการทดลองใหม่

## โครงสร้าง dataset

ใช้โฟลเดอร์เดิมตามแผน โดย `valid` เป็นโฟลเดอร์ validation และ key ใน YAML ใช้ `val`:

```text
pre operator internship/
├── data.yaml
├── train/
│   ├── images/   # .jpg / .jpeg / .png
│   └── labels/   # ชื่อเดียวกับภาพ เปลี่ยนนามสกุลเป็น .txt
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

หนึ่งบรรทัดใน label เป็น `class_id x_center y_center width height` ใช้พิกัด normalized 0–1 ไฟล์ว่างใช้แทนภาพพื้นหลังที่ไม่มีป้ายกำกับวัตถุได้ การตรวจรูปแบบไม่ได้ยืนยันว่าผู้ทำ annotation ใส่ตำหนิครบหรือถูกประเภท

| ID | คลาส |
|---|---|
| 0 | missing_hole |
| 1 | mouse_bite |
| 2 | open_circuit |
| 3 | short |
| 4 | spur |
| 5 | spurious_copper |

`data.yaml` กำหนด path จากโฟลเดอร์ YAML สคริปต์ของโครงการแปลงเป็น absolute path ก่อนส่งให้ Ultralytics จึงควรใช้คำสั่งผ่าน `python -m scripts...` จากโฟลเดอร์โครงการ

## ตรวจข้อมูลโดยไม่ฝึก

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_dataset
```

เขียนผลตรวจใน [reports/dataset_audit.json](reports/dataset_audit.json) รวมรายชื่อไฟล์ hash ป้ายกำกับ สถิติคลาส ภาพซ้ำข้าม split และกลุ่มต้นทางที่คาดจากชื่อ Roboflow คำสั่งนี้ไม่แก้ภาพ label หรือการแบ่งชุดข้อมูล

ผลตรวจปัจจุบัน:

| Split | ภาพ | Label | ไฟล์ label ว่าง |
|---|---:|---:|---:|
| train | 5,810 | 5,810 | 1,758 |
| valid | 556 | 556 | 184 |
| test | 276 | 276 | 90 |
| รวม | 6,642 | 6,642 | 2,032 |

ไม่พบภาพเปิดไม่ได้หรือป้ายกำกับผิดรูปแบบ พบภาพที่มี SHA-256 เหมือนกันข้าม split **39 กลุ่ม รวม 878 ไฟล์** และจำนวนภาพน้อยกว่ารายงาน/export 10 ภาพ จึงต้องตรวจต้นทางและข้อจำกัดก่อนนำผลประเมินไปตีความ โค้ดตรวจได้เฉพาะไฟล์เหมือนกันและชื่อกลุ่ม ไม่รับรองว่าตรวจพบภาพคล้ายกันหรือ PCB ชิ้นเดียวกันครบทุกกรณี

## เตรียมโค้ดฝึกโดยยังไม่เริ่มเทรน

ค่าฝึกอยู่ที่ [configs/train.yaml](configs/train.yaml) ค่าเริ่มต้นใช้ YOLO11n, 640 px, CUDA GPU 0, batch 16, 100 epochs และ seed 42 ทั้งหมดเป็นค่าตั้งต้นเพื่อทดลองภายหลัง

### NVIDIA CUDA setup (PowerShell)

Install the CUDA build into the project environment, then verify GPU access:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

This uses the CUDA 12.6 PyTorch wheels and requires a compatible NVIDIA driver. The training config selects GPU `0`; change `device` to `cpu` for CPU training. Batch size 16 and full precision (`amp: false`) are the initial settings for the local 6 GB GPU. If GPU memory runs out, reduce `batch` to 12 or 8. Compare epoch times to check whether the larger batch improves throughput.

Run training from the project root. The note records the known duplicate images across dataset splits, which limit evaluation independence:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs/train.yaml --execute --leakage-note "Learning experiment; duplicate images exist across splits, evaluation is not independent"
```

```powershell
.\.venv\Scripts\python.exe -m scripts.train
```

คำสั่งปกติเป็น **dry run** ตรวจข้อมูลและสร้าง [reports/training_plan.json](reports/training_plan.json) โดยไม่ import โมเดล ไม่ดาวน์โหลดน้ำหนัก และไม่เรียก `model.train()` น้ำหนักตั้งต้น `models/pretrained/yolo11n.pt` ดาวน์โหลดและตรวจโหลดแล้ววันที่ 15 ตุลาคม 2026 แต่ยังต้องทบทวนภาพซ้ำข้าม split แผนที่สร้างก่อนดาวน์โหลดอาจยังแสดงว่าไฟล์น้ำหนักขาด ให้รัน dry run ใหม่เมื่อต้องการอัปเดตแผน

ขั้นตอนเมื่อจะฝึกในภายหลัง:

1. เตรียมน้ำหนักตั้งต้น YOLO11 จากแหล่ง Ultralytics และวางใน `models/pretrained/` หรือแก้ path ใน config
2. ตรวจผล dataset audit หากจัดชุดข้อมูลใหม่ ให้เก็บภาพต้นทางไว้และบันทึก manifest ที่แก้แล้ว
3. ปรับ device, batch และค่าทดลองตามเครื่อง แล้วรัน dry run ซ้ำ
4. เริ่มฝึกด้วย `--execute` เท่านั้น ถ้ายังใช้ split ที่มีภาพซ้ำ ต้องระบุเหตุผล/ข้อจำกัดที่ทบทวนแล้วผ่าน `--leakage-note`

ตัวอย่างคำสั่งสำหรับอนาคต ซึ่งยังไม่ได้เรียกในงานนี้:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs/train.yaml --execute --leakage-note "Reviewed duplicate groups; learning experiment only, evaluation is not independent"
```

สร้าง experiment แยกตาม timestamp พร้อม config, เวอร์ชันไลบรารี, dataset audit/manifest, น้ำหนัก และกราฟของ Ultralytics ไม่ประเมิน test หรือส่งน้ำหนักเข้าแอปอัตโนมัติ

## เปรียบเทียบ freeze กับไม่ freeze รอบละ 60 epochs

รันคำสั่งเดียวเพื่อฝึกต่อกันบน GPU เดียว: รอบแรก `freeze: 2` ล็อกโมดูล 0–1 (Conv สองโมดูลแรก) ตลอดรอบฝึก (ไม่มีการปลดอัตโนมัติ) รอบสอง `freeze: 0` ทั้งสองเริ่มจาก pretrained เดียวกัน ใช้ seed เดียวกัน, batch 8 และ early stopping (`patience: 15`) ฝึกสูงสุดรอบละ 60 epochs หยุดเมื่อคะแนน validation ไม่ดีขึ้นต่อเนื่อง 15 epochs แต่ละรอบอาจหยุดคนละ epoch สรุปผลใช้จำนวน epoch ที่ฝึกจริง และเริ่มรอบถัดไปอัตโนมัติเมื่อรอบแรกหยุดตามปกติ ค่าฝึกอื่นอ่านจาก `configs/train.yaml` โดยไม่แก้ไฟล์หลัก

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_freeze --execute
```

รันโดยไม่ใส่ `--execute` เพื่อตรวจแผนอย่างเดียว ผลอยู่ใน `experiments/freeze_comparison_<timestamp>/` แยกสองรอบใน `runs/` พร้อมโมเดล, TensorBoard และ CSV รายคลาส สรุป `comparison.csv` หลังแต่ละรอบมีคะแนนจาก epoch ที่ mAP50-95 สูงสุด, คะแนนสุดท้าย, เวลาเฉลี่ยต่อ epoch และเวลารวม ค่าที่ขึ้นต้น `best_` ทั้งหมดมาจาก epoch เดียวกัน ไม่ใช่ค่าสูงสุดของแต่ละ metric แยกกัน

หากรอบใดล้มเหลวจะหยุดก่อนเริ่มรอบถัดไป หากมีการเทรนอื่นอยู่ ให้รอจบก่อนเริ่มคำสั่งนี้ เปรียบเทียบความแม่นยำด้วย validation mAP50-95 และพิจารณาความเร็วแยกกัน ยังไม่ใช้ test ตัดสิน และการทดลอง seed เดียวพร้อมภาพซ้ำข้าม split ยังไม่ยืนยันผลกับภาพใหม่จริง

## ดูกราฟการเทรนด้วย TensorBoard

ติดตั้ง dependency สำหรับงานฝึกครั้งแรก:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

`configs/train.yaml` เปิด `tensorboard: true` ไว้แล้ว แต่ละการฝึกด้วย `--execute` จะเขียนไฟล์ `events.out.tfevents.*` ในโฟลเดอร์ experiment ของรอบนั้น เช่น `experiments/pcb_yolo11n_<timestamp>/` จึงเก็บกราฟแต่ละรอบแยกจากกัน พร้อมบันทึกตำแหน่ง log ใน `experiment.json`

เปิด PowerShell อีกหน้าหนึ่งที่โฟลเดอร์โปรเจกต์ แล้วรัน:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir experiments --host localhost --port 6006
```

เปิด <http://localhost:6006> แล้วเลือก run ในแถบด้านซ้ายเพื่อเปรียบเทียบกราฟ Loss ของ train/validation, Precision, Recall, mAP@0.5, mAP@0.5:0.95 และ Learning Rate กราฟเริ่มมีข้อมูลเมื่อการฝึกผ่าน epoch ที่มีผลแล้ว ใช้ Ctrl+C ในหน้าต่าง TensorBoard เพื่อหยุดเซิร์ฟเวอร์ โดยไม่หยุดกระบวนการฝึกที่อยู่อีกหน้าต่าง

รอบฝึกที่เริ่มด้วยโค้ดใหม่นี้บันทึกค่าจาก validation แยกคลาสทุก epoch ที่ **Scalars → per_class → ชื่อคลาส** ได้แก่ `precision`, `recall`, `f1`, `AP50`, `AP50_95` และเขียน `per_class_metrics.csv` ในโฟลเดอร์ experiment เดียวกัน แม้ปิด TensorBoard ก็ยังเก็บ CSV ไว้ เลือก epoch 10, 20, … เพื่อนำไปทำตารางย้อนหลังได้ ส่วน loss รวมยังอยู่ใน `results.csv` และกราฟเดิม

ค่า Precision, Recall และ F1 ใช้จุด confidence ที่ Ultralytics เลือกจาก validation ไม่ใช่ confidence ของแอป คลาสที่ไม่มี ground truth ใน validation จะไม่สร้างคะแนนแทนขึ้นมา การตรวจ `best.pt` ตอนจบไม่เขียนทับค่าของ epoch สุดท้าย การแก้โค้ดไม่มีผลกับ process ที่กำลังเทรนอยู่และไม่เติมค่ารายคลาสให้ log เก่า ต้องเริ่มรอบใหม่เพื่อเก็บประวัตินี้

หากต้องการดูรอบเดียว ให้เปลี่ยน `--logdir` เป็นโฟลเดอร์ของรอบนั้น ปิดการเก็บ log ได้ด้วย `tensorboard: false` การทำ dry run ไม่สร้าง event log และการเปิด TensorBoard ไม่ได้เริ่มการฝึกเอง กราฟย้อนหลังของรอบที่ไม่ได้เก็บ TensorBoard log จะไม่ปรากฏอัตโนมัติ

ใช้ scalar logging ของ [Ultralytics TensorBoard integration](https://docs.ultralytics.com/integrations/tensorboard/) และ flush/close writer เมื่อฝึกจบหรือถูกขัดจังหวะ

## ประเมินและส่งโมเดลเข้าแอปภายหลัง

ประเมิน validation เพื่อศึกษาและปรับค่า จากนั้นตรึงค่าแล้วจึงใช้ test ในขั้นสุดท้าย ทั้งการประเมินและการฝึกมี dry run เป็นค่าเริ่มต้น:

```powershell
# ตรวจแผนประเมินอย่างเดียว
.\.venv\Scripts\python.exe -m scripts.evaluate --model "experiments/<run>/weights/best.pt"

# รันจริงในภายหลังเมื่อมีโมเดล
.\.venv\Scripts\python.exe -m scripts.evaluate --model "experiments/<run>/weights/best.pt" --split val --execute --leakage-note "<ผลการทบทวนข้อมูลซ้ำ>"
```

รายงาน evaluation มี Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, ผลรายคลาส, กราฟ Confusion Matrix/PR ของตัวตรวจสอบ, เวลาเฉลี่ย/มัธยฐาน และตัวอย่าง FP/FN ค่าจาก validator ใช้ Confidence ต่ำเพื่อสร้าง PR/AP ส่วนตัวอย่างผิดพลาดและ benchmark ใช้ค่าตรวจจาก `--conf` จึงไม่ควรตีความว่าเป็นค่า operating point เดียวกัน

ภาพตัวอย่าง error ใช้ **สีแดงสำหรับ FP** และ **สีน้ำเงินสำหรับ FN** ภายใต้การจับคู่ชนิดเดียวกันแบบหนึ่งต่อหนึ่งที่ IoU 0.5 เป็นการวิเคราะห์ประกอบ ไม่ใช่การแทนวิธีคำนวณ AP ของ Ultralytics การวัด benchmark รวม decode และ predict ไม่รวม UI การวาดกรอบหรือบันทึกหลักฐาน

ส่งออกชุดโมเดลที่เลือกแล้ว:

```powershell
.\.venv\Scripts\python.exe -m scripts.export_model --weights "experiments/<run>/weights/best.pt" --experiment "experiments/<run>/experiment.json" --model-id "pcb-yolo11n-v1"
```

สร้าง `best.pt` และ `metadata.json` ใน `models/pcb/` ตรวจคลาส สถาปัตยกรรม และ checksum หากมีชุดโมเดลเดิมแล้ว ให้ใช้ `--output models/pcb-v2` เพื่อเก็บรุ่นเก่าไว้ แล้วเปิดแอปด้วย `--model models/pcb-v2/best.pt`

## หลักฐาน ประวัติ และการสำรองข้อมูล

- ภาพที่พบตำหนิบันทึกเป็น `outputs/pcb/YYYY/MM/DD/YYYYMMDD_HHMMSS_<event_id>_annotated.png`
- Metadata และกรอบทั้งหมดอยู่ใน `data/pcb_detections.db` พร้อมรุ่นโมเดล hash และค่าตรวจที่ใช้
- ไม่บันทึกภาพที่ไม่พบกรอบผ่านเกณฑ์ ผลจำลอง หรือผลที่ประมวลผลล้มเหลว
- หากบันทึกล้มเหลว ผลตรวจยังแสดงบนหน้าจอและลองบันทึกผลเดิมซ้ำได้ โดยไม่เพิ่มเหตุการณ์ซ้ำ
- สำรองฐานข้อมูลและ `outputs/pcb/` พร้อมกันขณะปิดแอป การย้ายไฟล์หลักฐานภายนอกอาจทำให้ประวัติเปิดภาพไม่ได้
- ข้อผิดพลาดอยู่ใน `logs/pcb.log` แอปตรวจไฟล์ที่หายหรือค้างตอนเริ่มทำงานและไม่ลบหลักฐานเอง

## ทดสอบซอฟต์แวร์

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

ชุดทดสอบอยู่ที่ `tests/` แยกจาก dataset `test/` ใช้ Qt แบบ offscreen ไม่ต้องเปิดหน้าต่างจริง มีกรณี transaction rollback, บันทึกซ้ำ, ภาพเสีย, EXIF, mapping/checksum, worker, Dialog, ประวัติ และ dry run รวมทั้งทดสอบ adapter กับ YOLO11 ที่สร้างน้ำหนักสุ่มชั่วคราวโดย **ไม่ฝึก** น้ำหนักทดสอบถูกลบพร้อมโฟลเดอร์ชั่วคราว และไม่ได้ส่งเข้าแอปสำหรับใช้งาน

ผลตรวจล่าสุดอยู่ใน [reports/validation.md](reports/validation.md) การทดสอบซอฟต์แวร์ไม่ได้ประเมินความแม่นยำของโมเดล PCB

## เอกสารและที่มา

[Requirement.md](Requirement.md), [design.md](design.md), [artitecture.md](artitecture.md) อธิบายขอบเขต แบบออกแบบ และโครงสร้างโค้ด ส่วน `overview.md` เป็นบันทึกเดิมเรื่องถนน ใช้เอกสาร PCB ทั้งสามเป็นฐานของต้นแบบนี้

ที่มาชุดข้อมูลและเครดิตคงไว้ใน `README.dataset.txt`, `README.roboflow.txt` และ `data.yaml` ตาม CC BY 4.0 ที่ระบุในข้อมูลส่งออก

อ้างอิง API ที่ใช้: [Ultralytics Train](https://docs.ultralytics.com/modes/train/), [Predict](https://docs.ultralytics.com/modes/predict/), [Validation](https://docs.ultralytics.com/modes/val/) และ source ของเวอร์ชันที่ติดตั้งใน environment นี้
