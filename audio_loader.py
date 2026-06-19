import os
import zipfile
import librosa

# 1. LOCATE THE MYSTERY FILE
zip_path = "data/archive_zip"
extract_path = "data/extracted_audio"  # We will extract everything here

print("🔍 Inspecting 'data/archive_zip'...")

# 2. CHECK IF IT'S A ZIP FILE
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print("📦 It's a ZIP archive! Extracting now...")
        zip_ref.extractall(extract_path)
        print(f"✅ Extraction complete! Files saved to: {extract_path}")
except zipfile.BadZipFile:
    print("❌ It's not a valid ZIP file.")
    print("🛠️ MANUAL FIX: Right-click 'archive_zip' in your 'data' folder,")
    print("   select 'Extract All...', and tell me the name of the extracted folder.")
    exit()
except FileNotFoundError:
    print("❌ The file 'data/archive_zip' doesn't exist!")
    print("   Did you move it?")
    exit()

# 3. AUTOMATICALLY FIND THE .WAV FILES (ANYWHERE inside the extracted folder)
audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
audio_files = []

print("\n🔎 Scanning extracted folders for audio files...")
for root, dirs, filenames in os.walk(extract_path):
    for f in filenames:
        if f.lower().endswith(audio_extensions):
            full_path = os.path.join(root, f)
            audio_files.append(full_path)
            print(f"   Found: {f}")

# 4. REPORT RESULTS
print(f"\n📊 TOTAL AUDIO FILES FOUND: {len(audio_files)}")

if audio_files:
    # Load the first one to prove it works
    audio, sr = librosa.load(audio_files[0], sr=16000)
    print(f"\n✅ SUCCESS! Loaded: {os.path.basename(audio_files[0])}")
    print(f"   Duration: {len(audio)/sr:.2f} seconds")
    print(f"   Sample Rate: {sr} Hz")
    
    # Save the list of all files for later
    with open("file_list.txt", "w") as f:
        for file in audio_files:
            f.write(file + "\n")
    print(f"\n📝 Saved full file list to 'file_list.txt'")
else:
    print("\n❌ Still no audio files found.")
    print("   Let's check what EXACTLY is inside 'data/extracted_audio':")
    if os.path.exists(extract_path):
        print("   Contents:", os.listdir(extract_path))
    else:
        print("   Extraction folder doesn't exist.")