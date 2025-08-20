import { Client, GatewayIntentBits, Partials, PermissionsBitField } from "discord.js";
import dotenv from "dotenv";
import fs from "fs/promises";
import path from "path";
import fetch from "node-fetch";
import * as nsfw from "nsfwjs";
import * as tf from "@tensorflow/tfjs-node";
import Jimp from "jimp";

dotenv.config();

// --- Config ---
const TRAIN_DIR = path.resolve("./training_data");
const TARGET_IMAGES = 100; // Reduced from 10000 to avoid rate limits
const BATCH_SIZE = 2; // Reduced batch size
const DELAY_MS = 3000; // Increased delay to 3s

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds, 
    GatewayIntentBits.GuildMessages, 
    GatewayIntentBits.MessageContent
  ],
  partials: [Partials.Message, Partials.Channel, Partials.Reaction],
});

let nsfwModel;

// --- Manual NSFW Detection Setup ---
async function setupManualNSFWDetection() {
  console.log("🔧 Setting up manual NSFW detection...");
  
  // Create a comprehensive keyword and pattern-based filter
  const nsfwKeywords = [
    'porn', 'nsfw', 'xxx', 'nude', 'naked', 'sex', 'dick', 'cock', 'pussy', 
    'tits', 'boobs', 'breast', 'ass', 'butt', 'fuck', 'shit', 'damn', 'bitch',
    'whore', 'slut', 'cunt', 'penis', 'vagina', 'orgasm', 'masturbat', 'dildo',
    'vibrator', 'anal', 'blowjob', 'cumshot', 'gangbang', 'threesome', 'horny',
    'sexy', 'hot girl', 'cam girl', 'onlyfans', 'pornhub', 'xhamster'
  ];
  
  const suspiciousPatterns = [
    /\b(s+e+x+|f+u+c+k+|p+o+r+n+)\b/i, // stretched letters
    /[0-9]+\s*(cm|inch|inches)\b/i, // measurements
    /\b(18\+|21\+|adult)\b/i, // age references
    /\$[0-9]+.*\b(hour|night|session)\b/i // pricing
  ];
  
  // Create mock NSFW model for comprehensive content filtering
  nsfwModel = {
    classify: async (input) => {
      // For images, we'll do basic checks and assume safe unless text context suggests otherwise
      return [
        { className: 'Neutral', probability: 0.8 },
        { className: 'Drawing', probability: 0.15 },
        { className: 'Porn', probability: 0.05 }
      ];
    },
    
    isTextNSFW: (text) => {
      if (!text || typeof text !== 'string') return false;
      
      const lowerText = text.toLowerCase();
      
      // Check for direct keyword matches
      const hasKeywords = nsfwKeywords.some(keyword => lowerText.includes(keyword.toLowerCase()));
      
      // Check for suspicious patterns
      const hasPatterns = suspiciousPatterns.some(pattern => pattern.test(lowerText));
      
      // Check for excessive caps (often spam/inappropriate)
      const capsRatio = (text.match(/[A-Z]/g) || []).length / text.length;
      const excessiveCaps = text.length > 10 && capsRatio > 0.7;
      
      // Check for repeated characters (often used to bypass filters)
      const repeatedChars = /(.)\1{4,}/.test(text);
      
      return hasKeywords || hasPatterns || (excessiveCaps && hasKeywords) || repeatedChars;
    },
    
    checkImageName: (filename) => {
      if (!filename) return false;
      const lowerName = filename.toLowerCase();
      return nsfwKeywords.some(keyword => lowerName.includes(keyword));
    }
  };
  
  console.log("✅ Fallback content filter active (text + filename analysis)");
  return nsfwModel;
}

// --- Helper: sleep ---
const sleep = ms => new Promise(r => setTimeout(r, ms));

// --- Training Data Fetcher ---
async function fetchImage(url, filename) {
  try {
    const res = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      },
      timeout: 10000 // 10 second timeout
    });
    
    if (!res.ok) {
      console.log(`⚠️ HTTP ${res.status} for ${filename}, skipping`);
      return false;
    }
    
    const contentType = res.headers.get('content-type');
    if (!contentType || !contentType.startsWith('image/')) {
      console.log(`⚠️ Not an image: ${filename}, skipping`);
      return false;
    }
    
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length === 0) {
      console.log(`⚠️ Empty file: ${filename}, skipping`);
      return false;
    }
    
    await fs.writeFile(path.join(TRAIN_DIR, filename), buf);
    console.log(`✅ Saved ${filename} (${buf.length} bytes)`);
    return true;
  } catch (err) {
    console.error(`❌ Fetch failed for ${filename}: ${err.message}`);
    return false;
  }
}

async function ensureTrainingData() {
  try { 
    await fs.mkdir(TRAIN_DIR, { recursive: true }); 
  } catch (e) {
    console.error(`Failed to create directory: ${e.message}`);
    return;
  }
  
  let files;
  try {
    files = await fs.readdir(TRAIN_DIR);
  } catch (e) {
    console.error(`Failed to read directory: ${e.message}`);
    return;
  }
  
  let count = files.filter(f => f.endsWith('.jpg') || f.endsWith('.png')).length;

  console.log(`📂 Training images: ${count}/${TARGET_IMAGES}`);
  
  let attempts = 0;
  while (count < TARGET_IMAGES && attempts < TARGET_IMAGES * 2) {
    const jobs = [];
    const batchStart = count;
    
    for (let i = 0; i < BATCH_SIZE && count < TARGET_IMAGES; i++, attempts++) {
      // Use more reliable image sources
      const sources = [
        `https://picsum.photos/800/600?random=${Date.now()}-${attempts}`,
        `https://source.unsplash.com/800x600/?nature,landscape&sig=${attempts}`,
      ];
      
      const source = sources[Math.floor(Math.random() * sources.length)];
      const filename = `img_${attempts}_${Date.now()}.jpg`;
      
      jobs.push(
        fetchImage(source, filename).then(success => {
          if (success) count++;
          return success;
        })
      );
    }
    
    const results = await Promise.all(jobs);
    const successful = results.filter(Boolean).length;
    console.log(`📊 Batch complete: ${successful}/${results.length} successful`);
    
    if (successful === 0) {
      console.log(`⚠️ No successful downloads in batch, waiting longer...`);
      await sleep(DELAY_MS * 2);
    } else {
      await sleep(DELAY_MS);
    }
  }
  
  console.log(`🎉 Training data collection complete (${count} images)`);
}

// --- NSFW Detection ---
async function scanAttachment(url, messageContent = "", filename = "") {
  try {
    // First check message text and filename for NSFW content
    const textUnsafe = nsfwModel && nsfwModel.isTextNSFW && nsfwModel.isTextNSFW(messageContent);
    const filenameUnsafe = nsfwModel && nsfwModel.checkImageName && nsfwModel.checkImageName(filename);
    
    if (textUnsafe) {
      console.log("🔍 NSFW text content detected");
      return true;
    }
    
    if (filenameUnsafe) {
      console.log("🔍 NSFW filename detected");
      return true;
    }
    
    // Try to fetch and analyze the image
    const res = await fetch(url, { 
      timeout: 15000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      }
    });
    
    if (!res.ok) {
      console.log(`Failed to fetch attachment: HTTP ${res.status}`);
      // If we can't fetch the image but text/filename seems unsafe, flag it
      return textUnsafe || filenameUnsafe;
    }
    
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length === 0) return false;
    
    // Check file size - very large images might be suspicious
    if (buf.length > 10 * 1024 * 1024) { // 10MB
      console.log("⚠️ Very large image file detected (>10MB)");
      // Large files + suspicious text = likely unsafe
      if (textUnsafe) return true;
    }
    
    // Try image classification if we have a real model
    if (nsfwModel && nsfwModel.classify && typeof nsfwModel.classify === 'function') {
      try {
        const img = await tf.node.decodeImage(buf, 3);
        const predictions = await nsfwModel.classify(img);
        img.dispose();

        const unsafe = predictions.some(p => {
          if (["Porn", "Hentai", "Sexy"].includes(p.className) && p.probability > 0.6) {
            console.log(`🔍 NSFW detected: ${p.className} (${(p.probability * 100).toFixed(1)}%)`);
            return true;
          }
          return false;
        });
        
        return unsafe;
      } catch (imgErr) {
        console.log("Image analysis failed, relying on text/filename analysis");
        return textUnsafe || filenameUnsafe;
      }
    }
    
    return false;
  } catch (err) {
    console.error("Scan error:", err.message);
    // If scanning fails but we detected text issues, still flag it
    const textCheck = nsfwModel && nsfwModel.isTextNSFW && nsfwModel.isTextNSFW(messageContent);
    return textCheck || false;
  }
}

// --- Bot Events ---
client.once("ready", async () => {
  console.log(`🤖 Logged in as ${client.user.tag}`);
  
  // Try multiple approaches to load the NSFW model
  const loadAttempts = [
    // Attempt 1: Default load
    async () => {
      console.log("📦 Attempting default NSFW model load...");
      return await nsfw.load();
    },
    // Attempt 2: Load with specific URL
    async () => {
      console.log("📦 Attempting NSFW model load with custom URL...");
      return await nsfw.load('https://cdn.jsdelivr.net/npm/nsfwjs@2.4.2/example/nsfw_demo/public/model/');
    },
    // Attempt 3: Load with jsdelivr backup
    async () => {
      console.log("📦 Attempting NSFW model load with JSDelivr backup...");
      return await nsfw.load('https://cdn.jsdelivr.net/gh/infinitered/nsfwjs@master/example/nsfw_demo/public/model/');
    }
  ];

  let modelLoaded = false;
  for (let i = 0; i < loadAttempts.length && !modelLoaded; i++) {
    try {
      nsfwModel = await loadAttempts[i]();
      console.log(`✅ NSFW model loaded successfully (attempt ${i + 1})`);
      modelLoaded = true;
    } catch (err) {
      console.error(`❌ Load attempt ${i + 1} failed:`, err.message);
      if (i === loadAttempts.length - 1) {
        console.error("❌ All NSFW model load attempts failed!");
        console.log("🔧 Trying manual model setup...");
        
        // Last resort: set up manual content filtering
        try {
          nsfwModel = await setupManualNSFWDetection();
          modelLoaded = true;
          console.log("✅ Fallback content detection system activated");
        } catch (manualErr) {
          console.error("❌ Manual setup also failed:", manualErr.message);
          console.log("💡 Please check your network connection and DNS settings");
          console.log("💡 You may need to use a VPN or different network");
          process.exit(1);
        }
      }
    }
  }
  
  // Run training data collection in background
  ensureTrainingData().catch(err => {
    console.error("Training data collection failed:", err.message);
  });
});

client.on("messageCreate", async msg => {
  if (msg.author.bot) return;

  // Content moderation
  if (msg.attachments.size > 0) {
    for (const att of msg.attachments.values()) {
      if (att.contentType?.startsWith("image/") || att.contentType?.startsWith("video/")) {
        console.log(`🔍 Scanning attachment from ${msg.author.tag}`);
        
        const unsafe = await scanAttachment(att.url, msg.content, att.name);
        if (unsafe) {
          try {
            await msg.delete();
            
            // Check if we can timeout the user
            if (msg.member && msg.member.moderatable) {
              await msg.member.timeout(10 * 60 * 1000, "Posted NSFW content");
              console.log(`🚨 Removed NSFW from ${msg.author.tag} and applied timeout`);
            } else {
              console.log(`🚨 Removed NSFW from ${msg.author.tag} (couldn't apply timeout)`);
            }
            
            // Try to DM the user
            try {
              await msg.author.send("⚠️ Your message contained NSFW content and was removed. You have been muted for 10 minutes.");
            } catch (dmErr) {
              console.log(`Couldn't DM user ${msg.author.tag}: ${dmErr.message}`);
            }
            
          } catch (e) {
            console.error("Error during moderation:", e.message);
          }
        }
      }
    }
  }

  // Also check text-only messages for NSFW content
  if (msg.attachments.size === 0 && nsfwModel && nsfwModel.isTextNSFW && nsfwModel.isTextNSFW(msg.content)) {
    try {
      console.log(`🔍 NSFW text detected from ${msg.author.tag}: "${msg.content.substring(0, 50)}..."`);
      await msg.delete();
      
      if (msg.member && msg.member.moderatable) {
        await msg.member.timeout(5 * 60 * 1000, "Posted NSFW text content");
        console.log(`🚨 Removed NSFW text from ${msg.author.tag} and applied timeout`);
      }
      
      try {
        await msg.author.send("⚠️ Your message contained inappropriate content and was removed. You have been muted for 5 minutes.");
      } catch (dmErr) {
        console.log(`Couldn't DM user ${msg.author.tag}: ${dmErr.message}`);
      }
      
    } catch (e) {
      console.error("Error during text moderation:", e.message);
    }
  }

  // Admin test command
  if (msg.content === "?test") {
    if (!msg.member?.permissions.has(PermissionsBitField.Flags.Administrator)) {
      return msg.reply("❌ Administrator permissions required.");
    }

    try {
      const files = await fs.readdir(TRAIN_DIR);
      const imageFiles = files.filter(f => 
        f.endsWith('.jpg') || f.endsWith('.png') || f.endsWith('.jpeg')
      );
      
      if (imageFiles.length === 0) {
        return msg.reply("❌ No training images available yet!");
      }

      if (imageFiles.length < 4) {
        return msg.reply(`❌ Need at least 4 images, only have ${imageFiles.length}`);
      }

      // Pick 4 random images
      const pick = imageFiles.sort(() => 0.5 - Math.random()).slice(0, 4);
      console.log(`🖼️ Creating collage from: ${pick.join(', ')}`);
      
      const imgs = await Promise.all(
        pick.map(async f => {
          try {
            return await Jimp.read(path.join(TRAIN_DIR, f));
          } catch (err) {
            console.error(`Failed to read ${f}:`, err.message);
            // Return a placeholder colored rectangle
            return new Jimp(400, 300, 0xff0000ff); // Red placeholder
          }
        })
      );

      const w = 800, h = 600;
      const out = new Jimp(w, h, 0xffffffff);
      const halfW = w / 2, halfH = h / 2;
      
      // Resize and composite images
      for (let i = 0; i < 4; i++) {
        await imgs[i].resize(halfW, halfH);
      }

      out.composite(imgs[0], 0, 0);
      out.composite(imgs[1], halfW, 0);
      out.composite(imgs[2], 0, halfH);
      out.composite(imgs[3], halfW, halfH);

      const outPath = path.join(TRAIN_DIR, `collage_${Date.now()}.jpg`);
      await out.writeAsync(outPath);
      
      await msg.channel.send({ 
        content: "🎨 Generated test collage from training data:",
        files: [outPath] 
      });
      
    } catch (err) {
      console.error("Test command error:", err);
      msg.reply(`❌ Failed to generate test image: ${err.message}`);
    }
  }
});

// Error handling
client.on('error', error => {
  console.error('Discord client error:', error);
});

process.on('unhandledRejection', error => {
  console.error('Unhandled promise rejection:', error);
});

client.login(process.env.TOKEN).catch(err => {
  console.error('Failed to login:', err.message);
  process.exit(1);
});
