import fs from 'node:fs';
import {chromium} from '/home/truongvu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=process.cwd(),out=root+'/artifacts/audio_studio_web_20261018/';
const fixture=JSON.parse(fs.readFileSync(out+'qa_fixture.json'));
const browser=await chromium.launch({headless:true,executablePath:'/home/truongvu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome',args:['--no-sandbox']});
try{
 const page=await browser.newPage({viewport:{width:390,height:844}}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:8010',{waitUntil:'networkidle'});
 const recorded=await page.evaluate(async bytes=>{
  const ctx=new AudioContext(),buf=await ctx.decodeAudioData(new Uint8Array(bytes).buffer),src=ctx.createBufferSource(),dest=ctx.createMediaStreamDestination();src.buffer=buf;src.connect(dest);
  const recorder=new MediaRecorder(dest.stream,{mimeType:'audio/webm;codecs=opus'}),chunks=[];
  return await new Promise((resolve,reject)=>{recorder.ondataavailable=e=>chunks.push(e.data);recorder.onerror=reject;recorder.onstop=async()=>{const data=await new Blob(chunks).arrayBuffer();await ctx.close();dest.stream.getTracks().forEach(t=>t.stop());resolve(Array.from(new Uint8Array(data)));};recorder.start();src.onended=()=>recorder.stop();src.start();});
 },Array.from(fs.readFileSync(fixture.wav)));
 fs.writeFileSync(out+'recording-test.webm',Buffer.from(recorded));
 await page.locator('#file').setInputFiles(out+'recording-test.webm');
 await page.waitForFunction(()=>!state.converting&&!!state.audio);
 const info=await page.evaluate(()=>({rate:state.audio.rate,channels:state.audio.channels,bits:state.audio.bits,duration:state.audio.duration,type:state.file.type,name:state.file.name}));
 if(info.rate!==16000||info.channels!==1||info.bits!==16||info.type!=='audio/wav')throw new Error('Invalid conversion '+JSON.stringify(info));
 const event=page.waitForEvent('download');await page.locator('#download-wav').click();const download=await event;await download.saveAs(out+'converted-recording.wav');
 const wav=fs.readFileSync(out+'converted-recording.wav');if(wav.toString('ascii',0,4)!=='RIFF'||wav.readUInt32LE(24)!==16000||wav.readUInt16LE(22)!==1||wav.readUInt16LE(34)!==16)throw new Error('Invalid downloadable WAV');
 await page.screenshot({path:out+'mobile-converted.png',fullPage:true});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw new Error('Mobile overflow');
 await page.locator('#confirm-upload').click();await page.locator('#run').click();
 await page.waitForFunction(()=>state.result||!document.querySelector('#notice').hidden,null,{timeout:120000});
 if(!await page.locator('#result-screen').isVisible())throw new Error(await page.locator('#notice').textContent());
 await page.locator('#new-audio').click();
 await page.locator('#file').setInputFiles({name:'broken.mp3',mimeType:'audio/mpeg',buffer:Buffer.from('invalid audio')});
 await page.waitForFunction(()=>!state.converting&&!document.querySelector('#notice').hidden);
 if(!await page.locator('#confirm-upload').isDisabled())throw new Error('Invalid audio accepted');
 if(errors.length)throw new Error(errors.join('\n'));
 fs.writeFileSync(out+'conversion_verification.json',JSON.stringify({passed:true,input:'WebM Opus generated with browser MediaRecorder',output:info,wav_header_verified:true,inference_completed:true,invalid_audio_rejected:true,mobile_horizontal_overflow:false},null,2));
 console.log('Recorded WebM -> mono 16kHz PCM16 WAV, download, real inference, mobile layout and corrupt input passed.');
}finally{await browser.close();}
