class OneEuroAxis {
  constructor(mc,b,dc){this.mc=mc;this.b=b;this.dc=dc;this.x=null;this.dx=0;this.t=null;}
  _a(c,dt){const tau=1/(2*Math.PI*c);return 1/(1+tau/dt);}
  filter(x,t){ if(this.x===null){this.x=x;this.t=t;return x;}
    const dt=Math.max(1e-3,(t-this.t)/1000); this.t=t;
    const raw=(x-this.x)/dt; this.dx+=this._a(this.dc,dt)*(raw-this.dx);
    this.x+=this._a(this.mc+this.b*Math.abs(this.dx),dt)*(x-this.x); return this.x; }
}
function rnd(s){let z=s>>>0;return()=>{z=(z*1664525+1013904223)>>>0;return z/4294967296;};}
function gauss(R){return Math.sqrt(-2*Math.log(R()+1e-12))*Math.cos(2*Math.PI*R());}
const FPS=30;
function evalCfg(mc,b,dc,SIG){
  let tre=0,n=0;
  for(const seed of [7,19,31,43]){
    const R=rnd(seed); const eu=new OneEuroAxis(mc,b,dc);
    for(let i=0;i<500;i++){ const e=eu.filter(500+gauss(R)*SIG,i*1000/FPS); if(i>80){tre+=(e-500)**2;n++;} }
  }
  const tremor=Math.sqrt(tre/n);
  let settle=0,sn=0;
  for(const seed of [11,23,37,53]){
    const R=rnd(seed); const eu=new OneEuroAxis(mc,b,dc); let s=-1;
    for(let i=0;i<260;i++){ const truth=i<30?200:900; const e=eu.filter(truth+gauss(R)*SIG,i*1000/FPS);
      if(i>30&&s<0&&Math.abs(e-900)<60) s=(i-30)*1000/FPS; }
    settle+=(s<0?2000:s); sn++;
  }
  return {tremor, settle:settle/sn};
}
// EMA baseline
function ema(SIG){
  let tre=0,n=0;
  for(const seed of [7,19,31,43]){ const R=rnd(seed); let e=null;
    for(let i=0;i<500;i++){ const raw=500+gauss(R)*SIG; e=e===null?raw:e+0.35*(raw-e); if(i>80){tre+=(e-500)**2;n++;} } }
  let settle=0,sn=0;
  for(const seed of [11,23,37,53]){ const R=rnd(seed); let e=null,s=-1;
    for(let i=0;i<260;i++){ const truth=i<30?200:900; const raw=truth+gauss(R)*SIG; e=e===null?raw:e+0.35*(raw-e);
      if(i>30&&s<0&&Math.abs(e-900)<60) s=(i-30)*1000/FPS; }
    settle+=(s<0?2000:s); sn++; }
  return {tremor:Math.sqrt(tre/n), settle:settle/sn};
}
const SIG=45;
const base=ema(SIG);
console.log(`baseline EMA a=0.35:  tremor ${base.tremor.toFixed(1)} px   settle ${base.settle.toFixed(0)} ms\n`);
console.log("  minCut   beta     dCut   tremor    settle    verdict");
let best=null;
for(const mc of [0.3,0.5,0.8,1.2])
  for(const b of [0.0002,0.0006,0.0015,0.004,0.010])
    for(const dc of [0.5,1.0]){
      const r=evalCfg(mc,b,dc,SIG);
      const betterBoth = r.tremor<base.tremor && r.settle<base.settle;
      if(betterBoth && (!best || r.tremor<best.r.tremor)) best={mc,b,dc,r};
      if(betterBoth) console.log(`  ${mc.toFixed(2)}   ${String(b).padEnd(7)} ${dc.toFixed(1)}   ${r.tremor.toFixed(1)} px   ${r.settle.toFixed(0)} ms   better on both`);
    }
console.log(`\nchosen: minCutoff=${best.mc} beta=${best.b} dCutoff=${best.dc}`);
console.log(`  tremor ${base.tremor.toFixed(1)} -> ${best.r.tremor.toFixed(1)} px  (${(100*(1-best.r.tremor/base.tremor)).toFixed(0)}% less)`);
console.log(`  settle ${base.settle.toFixed(0)} -> ${best.r.settle.toFixed(0)} ms  (${(100*(1-best.r.settle/base.settle)).toFixed(0)}% faster)`);
