if(!self.define){const e=e=>{"require"!==e&&(e+=".js");let s=Promise.resolve();return a[e]||(s=new Promise((async s=>{if("document"in self){const a=document.createElement("script");a.src=e,document.head.appendChild(a),a.onload=s}else importScripts(e),s()}))),s.then((()=>{if(!a[e])throw new Error(`Module ${e} didn’t register its module`);return a[e]}))},s=(s,a)=>{Promise.all(s.map(e)).then((e=>a(1===e.length?e[0]:e)))},a={require:Promise.resolve(s)};self.define=(s,f,d)=>{a[s]||(a[s]=Promise.resolve().then((()=>{let a={};const r={uri:location.origin+s.slice(1)};return Promise.all(f.map((s=>{switch(s){case"exports":return a;case"module":return r;default:return e(s)}}))).then((e=>{const s=d(...e);return a.default||(a.default=s),a}))})))}}define("./service-worker.js",["./workbox-cab25c8f"],(function(e){"use strict";e.setCacheNameDetails({prefix:"mr-hope"}),self.addEventListener("message",(e=>{e.data&&"SKIP_WAITING"===e.data.type&&self.skipWaiting()})),e.clientsClaim(),e.precacheAndRoute([{url:"assets/css/0.styles.d02f10b1.css",revision:"b3acc4d53d3e9d9b97fd54fd19e37c1e"},{url:"assets/img/danger-dark.5fb01a54.svg",revision:"5fb01a54b1893112ce78f6bf6fe7b9b6"},{url:"assets/img/danger.6e8b05e0.svg",revision:"6e8b05e040e31db4b52092926ac89628"},{url:"assets/img/default-skin.b257fa9c.svg",revision:"b257fa9c5ac8c515ac4d77a667ce2943"},{url:"assets/img/fast-rcnn.8e472d35.svg",revision:"8e472d35d37e07884f68c1f96aa25868"},{url:"assets/img/faster-rcnn.29666caa.svg",revision:"29666caaa1c656419c6a7d3f19b9aaff"},{url:"assets/img/info-dark.9cb8f08f.svg",revision:"9cb8f08f92e9e47faf596484d9af636a"},{url:"assets/img/info.b3407763.svg",revision:"b3407763d94949efc3654309e9a2202f"},{url:"assets/img/r-cnn.3213f72c.svg",revision:"3213f72ceb2f4c4ad364d60cf7334cc6"},{url:"assets/img/search.83621669.svg",revision:"83621669651b9a3d4bf64d1a670ad856"},{url:"assets/img/tip-dark.36f60759.svg",revision:"36f607593c4f5274775cb24f1edf4389"},{url:"assets/img/tip.fa255ccc.svg",revision:"fa255cccbbef66519a1bb90a5bed6f24"},{url:"assets/img/warning-dark.34208d06.svg",revision:"34208d0652668027b5e1797bdafd2da9"},{url:"assets/img/warning.950d1128.svg",revision:"950d1128cc862f2b631f5d54c3458174"},{url:"assets/js/29.5660c1ad.js",revision:"d6d36f37a77ef80ed38cddf1959c4c5e"},{url:"assets/js/30.a4213f1d.js",revision:"9f8ff161b6b2f31a804a488f7cc9f162"},{url:"assets/js/31.47426dfc.js",revision:"ba76b5405c91b88d6e1585f6a9c9644e"},{url:"assets/js/32.5922ca09.js",revision:"36f264866ca573c7dd3179fb68e16fea"},{url:"assets/js/33.0bdbd4b1.js",revision:"82dc411967156513a807dfef4d2c373f"},{url:"assets/js/app.3d247a0d.js",revision:"373cd9c1f28015c4117ad07dff86e29c"},{url:"assets/js/layout-Blog.dd4a30c7.js",revision:"c9d8b4fe79f32945d7a156c910b859c9"},{url:"assets/js/layout-Layout.8002b20c.js",revision:"10d34970f786b1cf9c9e21a6ea36028e"},{url:"assets/js/layout-NotFound.67490558.js",revision:"8178e576a2dbd67b3a7849c6dfab7d83"},{url:"assets/js/layout-Slide.6b744727.js",revision:"7e05d972a1d56a9665141a9a49799452"},{url:"assets/js/page--3377aca7.19200a66.js",revision:"75e3cf52ad39647cd223bb86476ac180"},{url:"assets/js/page-记录生活.f699a647.js",revision:"69ef5580be9ef490ee4f3c06a748afe7"},{url:"assets/js/page-目标检测常用网络笔记.b86eb8be.js",revision:"958090c3de9fc910efc374e5ea3c69f4"},{url:"assets/js/page-算法刷题笔记.12e1b354.js",revision:"8d66f94cb63f5af1442aa177b7325e97"},{url:"assets/js/page-信号的采样.87f8a39a.js",revision:"def8ca580b546eecb254dc7b632f7fbd"},{url:"assets/js/page-d2l目标检测笔记.9c279315.js",revision:"bd8d88456f4fc0eed1843e8bddac8b5c"},{url:"assets/js/page-DSP,yyds!.4ce63113.js",revision:"8c7a614b1558be1d68c2487f9775a3a3"},{url:"assets/js/page-FFT.2c444c07.js",revision:"4f089f9766539c9f390ed52de068a4fd"},{url:"assets/js/page-hereisNYC.fba4e5fc.js",revision:"95f167f79a4db2b9d96bb0d57ff39254"},{url:"assets/js/page-Iloveboston!.73df5c0a.js",revision:"fe61e62ea0114afa21531594f04294fc"},{url:"assets/js/page-LDA.fdf61c1b.js",revision:"e57c8ade036b0e2bfa2684cd046d6d9c"},{url:"assets/js/page-Leetcode-5691通过最少操作次数使数组的和相等.56729c11.js",revision:"c517ef6bb3d7789f2a111988f549b85a"},{url:"assets/js/page-Mytour.cd7fc859.js",revision:"666757b644f00ee7109c6d464a27a5f1"},{url:"assets/js/page-notes!.a31f382a.js",revision:"d9cbe2e28319aa979e39ae0a78c8a130"},{url:"assets/js/page-PatternRecognition.4f6f5a3a.js",revision:"20a895e6ac39e9589bcf360bf69fe054"},{url:"assets/js/page-Report!.8ac7fb57.js",revision:"441aa83f52f7f5b2a227befb6edc13ad"},{url:"assets/js/page-WelcometoCV!.ead4d58e.js",revision:"310656b974d5822cdcb8db4ecf99522e"},{url:"assets/js/vendors~flowchart.225024d6.js",revision:"a40bb2bae513fe6c3427a0227277187f"},{url:"assets/js/vendors~layout-Blog~layout-Layout.8d7dc64f.js",revision:"d646c45a2aedc1cd3fe32fc95bbf2fa0"},{url:"assets/js/vendors~layout-Blog~layout-Layout~layout-NotFound.cfcaf0fe.js",revision:"6fa48f45598c847273d1f33265cdf16c"},{url:"assets/js/vendors~layout-Blog~layout-Layout~layout-NotFound~layout-Slide.817296dc.js",revision:"e91702f7becb16a19536fb51b35c6a71"},{url:"assets/js/vendors~photo-swipe.4013e7eb.js",revision:"ebe3049abbda3e7c40a91a9442f1b0d3"},{url:"assets/js/vendors~reveal.42050ef4.js",revision:"6216f00e1b0dd502a6c7eed14374dac7"},{url:"W.svg",revision:"cd83b8fdd2ac68c53cda43a2b82a2b61"},{url:"assets/fonts/KaTeX_AMS-Regular.2dbe16b4.ttf",revision:"2dbe16b4f4662798159f8d62c8d2509d"},{url:"assets/fonts/KaTeX_AMS-Regular.38a68f7d.woff2",revision:"38a68f7d18d292349a6e802a66136eae"},{url:"assets/fonts/KaTeX_AMS-Regular.7d307e83.woff",revision:"7d307e8337b9559e4040c5fb76819789"},{url:"assets/fonts/KaTeX_Caligraphic-Bold.33d26881.ttf",revision:"33d26881e4dd89321525c43b993f136c"},{url:"assets/fonts/KaTeX_Caligraphic-Regular.5e7940b4.ttf",revision:"5e7940b4ed250e98a512f520e39c867d"},{url:"assets/fonts/KaTeX_Fraktur-Bold.4de87d40.woff",revision:"4de87d40f0389255d975c69d45a0a7e7"},{url:"assets/fonts/KaTeX_Fraktur-Bold.7a3757c0.woff2",revision:"7a3757c0bfc580d91012d092ec8f06cb"},{url:"assets/fonts/KaTeX_Fraktur-Bold.ed330126.ttf",revision:"ed330126290a846bf0bb78f61aa6a080"},{url:"assets/fonts/KaTeX_Fraktur-Regular.450cc4d9.woff2",revision:"450cc4d9319c4a438dd00514efac941b"},{url:"assets/fonts/KaTeX_Fraktur-Regular.82d05fe2.ttf",revision:"82d05fe2abb0da9d1077110efada0f6e"},{url:"assets/fonts/KaTeX_Fraktur-Regular.dc4e330b.woff",revision:"dc4e330b6334767a16619c60d9ebce8c"},{url:"assets/fonts/KaTeX_Main-Bold.2e1915b1.ttf",revision:"2e1915b1a2f33c8ca9d1534193e934d7"},{url:"assets/fonts/KaTeX_Main-Bold.62c69756.woff",revision:"62c69756b3f1ca7b52fea2bea1030cd2"},{url:"assets/fonts/KaTeX_Main-Bold.78b0124f.woff2",revision:"78b0124fc13059862cfbe4c95ff68583"},{url:"assets/fonts/KaTeX_Main-BoldItalic.0d817b48.ttf",revision:"0d817b487b7fc993bda7dddba745d497"},{url:"assets/fonts/KaTeX_Main-BoldItalic.a2e3dcd2.woff",revision:"a2e3dcd2316f5002ee2b5f35614849a8"},{url:"assets/fonts/KaTeX_Main-BoldItalic.c7213ceb.woff2",revision:"c7213ceb79250c2ca46cc34ff3b1aa49"},{url:"assets/fonts/KaTeX_Main-Italic.081073fd.woff",revision:"081073fd6a7c66073ad231db887de944"},{url:"assets/fonts/KaTeX_Main-Italic.767e06e1.ttf",revision:"767e06e1df6abd16e092684bffa71c38"},{url:"assets/fonts/KaTeX_Main-Italic.eea32672.woff2",revision:"eea32672f64250e9d1dfb68177c20a26"},{url:"assets/fonts/KaTeX_Main-Regular.689bbe6b.ttf",revision:"689bbe6b67f22ffb51b15cc6cfa8facf"},{url:"assets/fonts/KaTeX_Main-Regular.756fad0d.woff",revision:"756fad0d6f3dff1062cfa951751d744c"},{url:"assets/fonts/KaTeX_Main-Regular.f30e3b21.woff2",revision:"f30e3b213e9a74cf7563b0c3ee878436"},{url:"assets/fonts/KaTeX_Math-BoldItalic.753ca3df.woff2",revision:"753ca3dfa44302604bec8e08412ad1c1"},{url:"assets/fonts/KaTeX_Math-BoldItalic.b3e80ff3.woff",revision:"b3e80ff3816595ffb07082257d30b24f"},{url:"assets/fonts/KaTeX_Math-BoldItalic.d9377b53.ttf",revision:"d9377b53f267ef7669fbcce45a74d4c7"},{url:"assets/fonts/KaTeX_Math-Italic.0343f93e.ttf",revision:"0343f93ed09558b81aaca43fc4386462"},{url:"assets/fonts/KaTeX_Math-Italic.2a39f382.woff2",revision:"2a39f3827133ad0aeb2087d10411cbf2"},{url:"assets/fonts/KaTeX_Math-Italic.67710bb2.woff",revision:"67710bb2357b8ee5c04d169dc440c69d"},{url:"assets/fonts/KaTeX_SansSerif-Bold.59b37733.woff2",revision:"59b3773389adfb2b21238892c08322ca"},{url:"assets/fonts/KaTeX_SansSerif-Bold.dfcc59ad.ttf",revision:"dfcc59ad994a0513b07ef3309b8b5159"},{url:"assets/fonts/KaTeX_SansSerif-Bold.f28c4fa2.woff",revision:"f28c4fa28f596796702fea3716d82052"},{url:"assets/fonts/KaTeX_SansSerif-Italic.3ab5188c.ttf",revision:"3ab5188c9aadedf425ea63c6b6568df7"},{url:"assets/fonts/KaTeX_SansSerif-Italic.99ad93a4.woff2",revision:"99ad93a4600c7b00b961d70943259032"},{url:"assets/fonts/KaTeX_SansSerif-Italic.9d0fdf5d.woff",revision:"9d0fdf5d7d27b0e3bdc740d90b40ec57"},{url:"assets/fonts/KaTeX_SansSerif-Regular.6c3bd5b5.woff",revision:"6c3bd5b57f7eba215a2d37e2e28077f1"},{url:"assets/fonts/KaTeX_SansSerif-Regular.badf3598.woff2",revision:"badf3598c22478fd9155a49391ecd396"},{url:"assets/fonts/KaTeX_SansSerif-Regular.d511ebce.ttf",revision:"d511ebcef253ab53775576f28315f350"},{url:"assets/fonts/KaTeX_Script-Regular.082640ca.ttf",revision:"082640ca4242bb2aade86855e4d7d5f6"},{url:"assets/fonts/KaTeX_Script-Regular.4edf4e0f.woff",revision:"4edf4e0fd49c8a5680dd541c05f16a4c"},{url:"assets/fonts/KaTeX_Script-Regular.af7bc98b.woff2",revision:"af7bc98b2200573686405dc784f53cf2"},{url:"assets/fonts/KaTeX_Size1-Regular.2c2dc3b0.ttf",revision:"2c2dc3b057bb48b80bc785ac3d87ecf8"},{url:"assets/fonts/KaTeX_Size2-Regular.114ad198.ttf",revision:"114ad19833311359052ad1a174159262"},{url:"assets/fonts/KaTeX_Size4-Regular.70174da7.ttf",revision:"70174da79d1707501c10e07872e84667"},{url:"assets/fonts/KaTeX_Typewriter-Regular.35fe2cce.ttf",revision:"35fe2cce0791c276b8e919decd873f5b"},{url:"assets/fonts/KaTeX_Typewriter-Regular.53dcf861.woff",revision:"53dcf861876aae6f3a6a6004dc3bc758"},{url:"assets/fonts/KaTeX_Typewriter-Regular.641339e2.woff2",revision:"641339e2cd86c7816bfb8028ee7f86e0"},{url:"404.html",revision:"4cb5714183c85a4948cefa546e423f00"},{url:"article/index.html",revision:"6ab5f4eb75c42b2400b56089211eeba7"},{url:"category/algorithm/index.html",revision:"dd01d5e42856dbae7e6fe47315efe301"},{url:"category/cv/index.html",revision:"d0a60ccea49106cf0b41fbc80d30785d"},{url:"category/dsp/index.html",revision:"13fa97de08acd26aeadbb62e29d30b98"},{url:"category/index.html",revision:"7428029f5ac9eda6f7e11d6a769ce8d1"},{url:"encrypt/index.html",revision:"2de73ffd852545d87b45aaabeecdd06e"},{url:"index.html",revision:"30c1d6bf4178d93c61885d7d2e785e7a"},{url:"life/index.html",revision:"6c84d1ece3d3038463dcd4fce3c96a49"},{url:"life/tour/bos/index.html",revision:"fad8e9df70531146933aa0313fc9111f"},{url:"life/tour/index.html",revision:"9c5d4baa525ba641df0d37f0993f5cd8"},{url:"life/tour/ny/index.html",revision:"b80e445c77399275922ff674da187961"},{url:"notes/algorithm/index.html",revision:"9e9b3623215b687785060235ac80abc6"},{url:"notes/algorithm/Leetcode-5691/index.html",revision:"0f65489cc7569d3668d6b2f707c6dc3a"},{url:"notes/cv/目标检测常用网络笔记/index.html",revision:"e2a74294e01b837e7593e7f23710b6b5"},{url:"notes/cv/d2l目标检测笔记/index.html",revision:"9ab2ed70779acf679f5ad6f61450f5fd"},{url:"notes/cv/index.html",revision:"355871738d5c73f8cb2e0afb0e1d94cd"},{url:"notes/index.html",revision:"923b31933ae0fa659925e0a75bafb6b2"},{url:"reports/dsp/信号的产生与采样/index.html",revision:"a595f96b183a46f8826db7382e521d75"},{url:"reports/dsp/fft/index.html",revision:"a97012123d7fc965c7f2db2ce0506fe3"},{url:"reports/dsp/index.html",revision:"2c5507ea4779ac0b8ff1112d27444b82"},{url:"reports/index.html",revision:"c299eb83194f0b464840271b07eddf3e"},{url:"reports/pr/ex1/index.html",revision:"e31dd1f533c4ae0595674df4f6aa7dfa"},{url:"reports/pr/index.html",revision:"52817eb3b0b9863f61f81742e350611e"},{url:"slide/index.html",revision:"8027cd6ddc7d7be1fcf6159c9c98146d"},{url:"tag/目标检测/index.html",revision:"5d4cacd30ad90614ea7b26c106c8593f"},{url:"tag/双指针/index.html",revision:"155bcac317b79e77f7cc21cd47e4aa26"},{url:"tag/dsp/index.html",revision:"70b570f93df545563e78f582519ba04c"},{url:"tag/fft/index.html",revision:"242d0f5a15f594241e3655ab72f60b10"},{url:"tag/index.html",revision:"3814aae047d5b06ebe51ea87df47e4d2"},{url:"tag/Leetcode/index.html",revision:"69d3ce409a445c7155e22534ed5c168b"},{url:"timeline/index.html",revision:"2a6c607f09d7fcdd482fbef604d85711"}],{}),e.cleanupOutdatedCaches()}));
//# sourceMappingURL=service-worker.js.map
addEventListener("message", (event) => {
  const replyPort = event.ports[0];
  const message = event.data;
  if (replyPort && message && message.type === "skip-waiting")
    event.waitUntil(
      self.skipWaiting().then(
        () => replyPort.postMessage({ error: null }),
        (error) => replyPort.postMessage({ error })
      )
    );
});
