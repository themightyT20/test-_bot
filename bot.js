// bot.js
import "dotenv/config";
import { Client, GatewayIntentBits, Partials } from "discord.js";
import * as tf from "@tensorflow/tfjs-node";
import nsfw from "nsfwjs";
import fs from "fs";
import path from "path";
import fetch from "node-fetch";
import { HfInference } from "@huggingface/inference";
import FurryDetector, { inferFurry } from "./furryDetector.js"; // ✅ removed getDetector

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
  partials: [Partials.Channel],
});

// KEEP YOUR mobilenet_v2 path exactly as you set it
const NSFW_MODEL_JSON = path.resolve("./mobilenet_v2/model.json");

let nsfwModel = null;
let hfClient = null;
let detector = null;
let isShuttingDown = false;

// === RAID PROTECTION CONFIG ===
const RAID_WINDOW_MS = 1000; // 1 second window
const RAID_THRESHOLD = 3; // more than 3 images in window triggers raid (i.e. 4+ calls)
const userImageTimestamps = new Map();

function noteImageAndCheckRaid(userId) {
  const now = Date.now();
  let arr = userImageTimestamps.get(userId);
  if (!arr) {
    arr = [];
    userImageTimestamps.set(userId, arr);
  }
  arr.push(now);
  while (arr.length && now - arr[0] > RAID_WINDOW_MS) {
    arr.shift();
  }
  return arr.length > RAID_THRESHOLD;
}

setInterval(() => {
  const now = Date.now();
  for (const [uid, arr] of userImageTimestamps.entries()) {
    while (arr.length && now - arr[0] > RAID_WINDOW_MS) arr.shift();
    if (arr.length === 0) userImageTimestamps.delete(uid);
  }
}, 60 * 1000);

// === Model loading ===
async function loadModels() {
  console.log("Loading NSFW model...");
  try {
    if (!fs.existsSync(NSFW_MODEL_JSON)) {
      throw new Error(`Model file not found at ${NSFW_MODEL_JSON}`);
    }
    nsfwModel = await nsfw.load(tf.io.fileSystem(NSFW_MODEL_JSON));
    console.log("✅ NSFW model loaded (mobilenet_v2)");
  } catch (err) {
    console.error("❌ Failed to load NSFW model:", err.message);
    nsfwModel = null;
  }

  if (process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN) {
    try {
      const key = process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN;
      hfClient = new HfInference(key);
      console.log("✅ Hugging Face client initialized");
    } catch (err) {
      console.warn("⚠️ Hugging Face client init failed:", err.message);
      hfClient = null;
    }
  } else {
    console.log("⚠️ No HF API key configured — fallback disabled");
  }

  // ✅ Just create detector directly (no getDetector)
  detector = new FurryDetector(hfClient);

  try {
    const ok = await detector.loadModel();
    if (ok) {
      console.log("✅ Furry model loaded (local)");
    } else {
      console.warn("⚠️ Local furry model not loaded — will use HF fallback when needed");
    }
  } catch (err) {
    console.warn("⚠️ Error initializing furry detector:", err.message);
  }
}

async function bufferFromUrl(url) {
  const res = await fetch(url, { timeout: 15000, headers: { "User-Agent": "DiscordBot/1.0" } });
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

async function detectNSFWContent(buffer) {
  let image, resized, batched;
  try {
    image = tf.node.decodeImage(buffer, 3);
    resized = tf.image.resizeBilinear(image, [224, 224]);
    batched = resized.expandDims(0).div(255.0);

    const nsfwPredictions = nsfwModel
      ? await nsfwModel.classify(batched)
      : [{ className: "Neutral", probability: 1 }];

    const nsfwContent = nsfwPredictions.some(
      (p) =>
        (p.className === "Porn" ||
          p.className === "Hentai" ||
          p.className === "Sexy") &&
        p.probability > 0.6
    );

    const maxNsfwProbability = Math.max(
      0,
      ...nsfwPredictions
        .filter((p) => ["Porn", "Hentai", "Sexy"].includes(p.className))
        .map((p) => p.probability)
    );

    return { isNSFW: nsfwContent, probability: maxNsfwProbability, predictions: nsfwPredictions };
  } catch (err) {
    console.error("❌ NSFW detection error:", err.message);
    return { isNSFW: false, probability: 0, predictions: [], error: err.message };
  } finally {
    if (image) image.dispose();
    if (resized) resized.dispose();
    if (batched) batched.dispose();
  }
}

// === Message handling ===
client.on("ready", () => {
  console.log(`🤖 Logged in as ${client.user.tag}`);
});

client.on("messageCreate", async (message) => {
  if (message.author.bot || isShuttingDown) return;
  if (!message.attachments || message.attachments.size === 0) return;

  try {
    const attachmentsArray = Array.from(message.attachments.values()).filter(att => att.contentType?.startsWith("image/"));
    if (attachmentsArray.length === 0) return;

    let raidDetected = false;
    for (let i = 0; i < attachmentsArray.length; i++) {
      if (noteImageAndCheckRaid(message.author.id)) {
        raidDetected = true;
        break;
      }
    }

    if (raidDetected) {
      await message.delete().catch(() => {});
      console.log(`🚨 RAID: Deleted message from ${message.author.tag}`);
      try {
        if (message.member?.moderatable) {
          await message.member.timeout(60 * 1000, "Raid detected: too many images");
          console.log(`⏱ Timed out ${message.author.tag} for raid`);
        }
      } catch {}
      return;
    }

    for (const attachment of attachmentsArray) {
      try {
        console.log(`🔍 Analyzing image from ${message.author.tag}`);
        const buffer = await bufferFromUrl(attachment.url);

        const nsfwResult = await detectNSFWContent(buffer);

        let furryResult = null;
        let usedHF = false;

        if (detector && detector.isLoaded) {
          try {
            const localRes = await detector.detectFurry(buffer);
            furryResult = localRes;
            console.log("🔍 Local furry detection result:", localRes);

            if (!localRes.isFurry && (detector.hf || hfClient)) {
              console.log("🔄 Local said SAFE — forcing HF fallback check...");
              const hfRes = detector.detectFurryWithHF
                ? await detector.detectFurryWithHF(buffer)
                : await inferFurry(buffer, hfClient);
              console.log("🔍 HF fallback result:", hfRes);
              usedHF = true;
              if (hfRes.isFurry) furryResult = hfRes;
            }
          } catch (err) {
            console.warn("⚠️ Local furry detection failed, using HF:", err.message);
            furryResult = await inferFurry(buffer, hfClient);
            usedHF = true;
          }
        } else {
          furryResult = await inferFurry(buffer, hfClient);
          usedHF = true;
        }

        console.log(`📊 Analysis Results:`);
        console.log(`   NSFW: ${nsfwResult.isNSFW ? "🔴 DETECTED" : "🟢 Safe"} (${(nsfwResult.probability * 100).toFixed(1)}%)`);
        if (furryResult) {
          const prob = furryResult.probability ?? 0;
          console.log(`   Furry: ${furryResult.isFurry ? "🔴 DETECTED" : "🟢 Safe"} (${(prob * 100).toFixed(1)}%) via ${usedHF ? "hf" : "local"}`);
        }

        if (nsfwResult.isNSFW || furryResult?.isFurry) {
          await message.delete().catch(() => {});
          console.log(`🚨 Deleted bad content from ${message.author.tag}`);
          if (message.member?.moderatable) {
            await message.member.timeout(10 * 60 * 1000, "Posted prohibited content");
          }
        } else {
          console.log("✅ Image appears safe");
        }
      } catch (err) {
        console.error(`❌ Error processing attachment: ${err.message}`);
      }
    }
  } catch (err) {
    console.error(`❌ Error processing images: ${err.message}`);
  }
});

client.on("error", (error) => console.error("❌ Discord client error:", error));
client.on("warn", (warning) => console.warn("⚠️ Discord client warning:", warning));

process.on("unhandledRejection", (error) => console.error("❌ Unhandled rejection:", error));
process.on("uncaughtException", (error) => { console.error("❌ Uncaught exception:", error); process.exit(1); });

process.on("SIGINT", async () => {
  console.log("\n🔄 Shutting down gracefully...");
  isShuttingDown = true;
  if (detector) await detector.shutdown();
  client.destroy();
  process.exit(0);
});

process.on("SIGTERM", async () => {
  console.log("🔄 Received SIGTERM, shutting down...");
  isShuttingDown = true;
  if (detector) await detector.shutdown();
  client.destroy();
  process.exit(0);
});

(async () => {
  try {
    await loadModels();
    await client.login(process.env.DISCORD_TOKEN);
  } catch (e) {
    console.error("❌ Startup error:", e.message);
    process.exit(1);
  }
})();
