/* Does the filter change actually reduce pointer tremor, and does dwell
 * hysteresis actually let a dwell complete near a key edge? */
const SMOOTHING = 0.35;
const EURO_MIN_CUTOFF = 0.30, EURO_BETA = 0.0006, EURO_DCUTOFF = 1.0;
class OneEuroAxis {
  constructor(){this.x=null;this.dx=0;this.t=null;}
  _alpha(c,dt){const tau=1/(2*Math.PI*c);return 1/(1+tau/dt);}
  filter(x,t){ if(this.x===null){this.x=x;this.t=t;return x;}
    const dt=Math.max(1e-3,(t-this.t)/1000); this.t=t;
    const raw=(x-this.x)/dt; this.dx+=this._alpha(EURO_DCUTOFF,dt)*(raw-this.dx);
    const cut=EURO_MIN_CUTOFF+EURO_BETA*Math.abs(this.dx);
    this.x+=this._alpha(cut,dt)*(x-this.x); return this.x; }
}
function rnd(s){let z=s>>>0;return()=>{z=(z*1664525+1013904223)>>>0;return z/4294967296;};}
function gauss(R){return Math.sqrt(-2*Math.log(R()+1e-12))*Math.cos(2*Math.PI*R());}

const FPS=30, SIG=Number(process.argv[2]||45);
// --- A. tremor while resting on a fixed target ---
{
  const R=rnd(7); let ema=null; const eu=new OneEuroAxis();
  let emaVar=0,euVar=0,n=0;
  for(let i=0;i<600;i++){
    const raw=500+gauss(R)*SIG, t=i*1000/FPS;
    ema = ema===null?raw:ema+SMOOTHING*(raw-ema);
    const e=eu.filter(raw,t);
    if(i>60){ emaVar+=(ema-500)**2; euVar+=(e-500)**2; n++; }
  }
  console.log(`Pointer tremor while resting (true position fixed, ${SIG} px sensor noise):`);
  console.log(`  fixed EMA a=0.35   ${Math.sqrt(emaVar/n).toFixed(1)} px RMS`);
  console.log(`  One-Euro adaptive  ${Math.sqrt(euVar/n).toFixed(1)} px RMS`);
  console.log(`  -> ${(100*(1-Math.sqrt(euVar/n)/Math.sqrt(emaVar/n))).toFixed(0)}% less tremor`);
}
// --- B. lag when the gaze crosses the screen ---
{
  const R=rnd(11); let ema=null; const eu=new OneEuroAxis();
  let emaSettle=-1, euSettle=-1;
  for(let i=0;i<200;i++){
    const truth = i<30 ? 200 : 900;
    const raw=truth+gauss(R)*SIG, t=i*1000/FPS;
    ema = ema===null?raw:ema+SMOOTHING*(raw-ema);
    const e=eu.filter(raw,t);
    if(i>30){
      if(emaSettle<0 && Math.abs(ema-900)<60) emaSettle=(i-30)*1000/FPS;
      if(euSettle<0 && Math.abs(e-900)<60) euSettle=(i-30)*1000/FPS;
    }
  }
  console.log(`\nSettling time after a 700 px saccade (within 60 px of target):`);
  console.log(`  fixed EMA          ${emaSettle.toFixed(0)} ms`);
  console.log(`  One-Euro adaptive  ${euSettle.toFixed(0)} ms`);
}
// --- C. dwell completion near a key edge, with and without hysteresis ---
{
  const KEY_W=140, DWELL_MS=693, MARGIN=20, EXIT_FRAMES=3;
  const need=Math.round(DWELL_MS/1000*FPS);
  function trial(seed, offset, hysteresis){
    const R=rnd(seed); const eu=new OneEuroAxis();
    let cur=null, start=0, out=0, frames=0;
    // key spans [0, KEY_W]; the user aims `offset` px from its right edge
    for(let i=0;i<200;i++){
      const raw=KEY_W-offset+gauss(R)*SIG, t=i*1000/FPS;
      const x=eu.filter(raw,t);
      let key = (x>=0 && x<=KEY_W) ? "A" : (x>KEY_W ? "B" : "C");
      if(hysteresis && key!==cur && cur){
        const inMargin = cur==="A" ? (x>=-MARGIN && x<=KEY_W+MARGIN) : false;
        if(inMargin){ out=0; key=cur; }
        else if(++out<EXIT_FRAMES) key=cur;
      } else if(hysteresis) out=0;
      if(key!==cur){ cur=key; start=i; frames=0; }
      else if(cur==="A"){ frames++; if(frames>=need) return true; }
    }
    return false;
  }
  console.log(`\nDwell completion on a ${KEY_W} px key, aiming N px inside its edge:`);
  console.log(`  offset   without hysteresis   with hysteresis`);
  for(const off of [70,40,25,15]){
    let a=0,b=0;
    for(let s=0;s<60;s++){ if(trial(s*7+1,off,false))a++; if(trial(s*7+1,off,true))b++; }
    console.log(`  ${String(off).padStart(3)} px      ${String((100*a/60).toFixed(0)+"%").padStart(8)}          ${String((100*b/60).toFixed(0)+"%").padStart(8)}`);
  }
}
