import io
import re

def remove_emojis():
    path = r"c:\Users\USER\Downloads\dropout_final\dropout\app.py"
    with io.open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Specific targeted replacements
    text = text.replace("❌ ", "")
    text = text.replace("💡 ", "")
    text = text.replace("⚠️ ", "")
    text = text.replace("📞 ", "")
    text = text.replace("💰 ", "")
    text = text.replace("📚 ", "")
    text = text.replace("🎯 ", "")
    text = text.replace("📋 ", "")
    text = text.replace("📊 ", "")
    text = text.replace("🤝 ", "")
    text = text.replace("💬 ", "")
    text = text.replace("🏆 ", "")
    text = text.replace("📜 ", "")
    text = text.replace("🌟 ", "")
    text = text.replace("⭐ ", "")
    text = text.replace("✅ ", "")
    text = text.replace("✅", "")
    text = text.replace("🔢 ", "")
    text = text.replace("🎓 ", "")
    text = text.replace("📌 ", "")
    text = text.replace("🔑 ", "")
    text = text.replace("🤖 ", "")
    text = text.replace("👥 ", "")
    text = text.replace("★", "")
    text = text.replace("⚠️", "Tidak")
    
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    remove_emojis()
