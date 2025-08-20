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
const TARGET_IMAGES = 10000;
const BATCH_SIZE = 5;
const DELAY_MS = 2000; // 2s between batches

const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent],
  partials: [Partials.Message, Partials.Channel, Partials.Reaction],
});

let nsfwModel;

// --- Helper: sleep ---
const sleep = ms => new Promise(r => setTimeout(r, ms));

// --- Training Data Fetcher ---
async function fetchImage(url, filename) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    await fs.writeFile(path.join(TRAIN_DIR, filename), buf);
    console.log(`✅ Saved ${filename}`);
  } catch (err) {
    console.error(`❌ Fetch failed: ${err.message}`);
  }
}

async function ensureTrainingData() {
  try { await fs.mkdir(TRAIN_DIR, { recursive: true }); } catch {}
  let files = await fs.readdir(TRAIN_DIR);
  let count = files.length;

  console.log(`📂 Training images: ${count}/${TARGET_IMAGES}`);
  while (count < TARGET_IMAGES) {
    const jobs = [];
    for (let i = 0; i < BATCH_SIZE && count < TARGET_IMAGES; i++, count++) {
      const source = [
        `https://source.unsplash.com/random/800x600?sig=${Date.now()}-${count}`,
        `https://picsum.photos/800/600?random=${count}`,
        `https://upload.wikimedia.org/wikipedia/commons/thumb/${Math.floor(Math.random()*10)}/00/Example.jpg/800px-Example.jpg`
      ][Math.floor(Math.random() * 3)];

      jobs.push(fetchImage(source, `img_${count}.jpg`));
    }
    await Promise.all(jobs);
    await sleep(DELAY_MS);
  }
  console.log(`🎉 Training data ready (${TARGET_IMAGES} images)`);
}

// --- NSFW Detection ---
async function scanAttachment(url) {
  try {
    const res = await fetch(url);
    const buf = Buffer.from(await res.arrayBuffer());
    const img = await tf.node.decodeImage(buf, 3);
    const predictions = await nsfwModel.classify(img);
    img.dispose();

    let unsafe = predictions.some(p => {
      if (["Porn", "Hentai", "Sexy"].includes(p.className) && p.probability > 0.7) return true;
      return false;
    });
    return unsafe;
  } catch (err) {
    console.error("Scan error:", err.message);
    return false;
  }
}

// --- Bot Events ---
client.once("ready", async () => {
  console.log(`🤖 Logged in as ${client.user.tag}`);
  nsfwModel = await nsfw.load();
  console.log("📦 NSFW model loaded");
  ensureTrainingData(); // runs in background
});

client.on("messageCreate", async msg => {
  if (msg.author.bot) return;

  // moderation
  if (msg.attachments.size > 0) {
    for (const att of msg.attachments.values()) {
      if (att.contentType?.startsWith("image") || att.contentType?.startsWith("video")) {
        const unsafe = await scanAttachment(att.url);
        if (unsafe) {
          try {
            await msg.delete();
            await msg.member.timeout(10 * 60 * 1000, "Posted NSFW content");
            await msg.author.send("⚠️ Your message contained NSFW content and was removed. You have been muted for 10 minutes.");
            console.log(`🚨 Removed NSFW from ${msg.author.tag}`);
          } catch (e) {
            console.error("Error deleting or muting:", e.message);
          }
        }
      }
    }
  }

  // admin command
  if (msg.content === "?test") {
    if (!msg.member.permissions.has(PermissionsBitField.Flags.Administrator)) {
      return msg.reply("❌ Admins only.");
    }

    // generate test image from dataset
    try {
      const files = await fs.readdir(TRAIN_DIR);
      if (files.length === 0) return msg.reply("No training images yet!");

      const pick = files.sort(() => 0.5 - Math.random()).slice(0, 4);
      const imgs = await Promise.all(pick.map(f => Jimp.read(path.join(TRAIN_DIR, f))));

      const w = 800, h = 600;
      const out = new Jimp(w, h, 0xffffffff);
      const halfW = w / 2, halfH = h / 2;
      await imgs[0].resize(halfW, halfH);
      await imgs[1].resize(halfW, halfH);
      await imgs[2].resize(halfW, halfH);
      await imgs[3].resize(halfW, halfH);

      out.composite(imgs[0], 0, 0);
      out.composite(imgs[1], halfW, 0);
      out.composite(imgs[2], 0, halfH);
      out.composite(imgs[3], halfW, halfH);

      const outPath = path.join(TRAIN_DIR, "test.jpg");
      await out.writeAsync(outPath);
      await msg.channel.send({ files: [outPath] });
    } catch (err) {
      console.error(err);
      msg.reply("❌ Failed to generate test image.");
    }
  }
});

client.login(process.env.TOKEN);
