if(!self.define){const e=e=>{"require"!==e&&(e+=".js");let a=Promise.resolve();return s[e]||(a=new Promise((async a=>{if("document"in self){const s=document.createElement("script");s.src=e,document.head.appendChild(s),s.onload=a}else importScripts(e),a()}))),a.then((()=>{if(!s[e])throw new Error(`Module ${e} didn’t register its module`);return s[e]}))},a=(a,s)=>{Promise.all(a.map(e)).then((e=>s(1===e.length?e[0]:e)))},s={require:Promise.resolve(a)};self.define=(a,f,d)=>{s[a]||(s[a]=Promise.resolve().then((()=>{let s={};const i={uri:location.origin+a.slice(1)};return Promise.all(f.map((a=>{switch(a){case"exports":return s;case"module":return i;default:return e(a)}}))).then((e=>{const a=d(...e);return s.default||(s.default=a),s}))})))}}define("./service-worker.js",["./workbox-cab25c8f"],(function(e){"use strict";e.setCacheNameDetails({prefix:"mr-hope"}),self.addEventListener("message",(e=>{e.data&&"SKIP_WAITING"===e.data.type&&self.skipWaiting()})),e.clientsClaim(),e.precacheAndRoute([{url:"assets/css/0.styles.d02f10b1.css",revision:"b3acc4d53d3e9d9b97fd54fd19e37c1e"},{url:"assets/img/danger-dark.5fb01a54.svg",revision:"5fb01a54b1893112ce78f6bf6fe7b9b6"},{url:"assets/img/danger.6e8b05e0.svg",revision:"6e8b05e040e31db4b52092926ac89628"},{url:"assets/img/default-skin.b257fa9c.svg",revision:"b257fa9c5ac8c515ac4d77a667ce2943"},{url:"assets/img/fast-rcnn.8e472d35.svg",revision:"8e472d35d37e07884f68c1f96aa25868"},{url:"assets/img/faster-rcnn.29666caa.svg",revision:"29666caaa1c656419c6a7d3f19b9aaff"},{url:"assets/img/info-dark.9cb8f08f.svg",revision:"9cb8f08f92e9e47faf596484d9af636a"},{url:"assets/img/info.b3407763.svg",revision:"b3407763d94949efc3654309e9a2202f"},{url:"assets/img/r-cnn.3213f72c.svg",revision:"3213f72ceb2f4c4ad364d60cf7334cc6"},{url:"assets/img/search.83621669.svg",revision:"83621669651b9a3d4bf64d1a670ad856"},{url:"assets/img/tip-dark.36f60759.svg",revision:"36f607593c4f5274775cb24f1edf4389"},{url:"assets/img/tip.fa255ccc.svg",revision:"fa255cccbbef66519a1bb90a5bed6f24"},{url:"assets/img/warning-dark.34208d06.svg",revision:"34208d0652668027b5e1797bdafd2da9"},{url:"assets/img/warning.950d1128.svg",revision:"950d1128cc862f2b631f5d54c3458174"},{url:"assets/js/36.7f3151ad.js",revision:"cafd19c14d0aaeb575ab51ada242daf4"},{url:"assets/js/37.129e2220.js",revision:"2ca3bcfc268bc3a43a92d24ef0795c22"},{url:"assets/js/38.b635d0aa.js",revision:"b2e8838354a543803fdde3008eef3099"},{url:"assets/js/39.da79145d.js",revision:"667a108359c42aad37b5e7e2fcefcffe"},{url:"assets/js/40.7fb14d47.js",revision:"a6b6bf42d21dd3a45ed65dfd7aba41fe"},{url:"assets/js/app.544f9747.js",revision:"c4176f0ea19af8e779a426aa966ffed6"},{url:"assets/js/layout-Blog.dd4a30c7.js",revision:"c9d8b4fe79f32945d7a156c910b859c9"},{url:"assets/js/layout-Layout.8002b20c.js",revision:"10d34970f786b1cf9c9e21a6ea36028e"},{url:"assets/js/layout-NotFound.67490558.js",revision:"8178e576a2dbd67b3a7849c6dfab7d83"},{url:"assets/js/layout-Slide.6b744727.js",revision:"7e05d972a1d56a9665141a9a49799452"},{url:"assets/js/page--3377aca7.19200a66.js",revision:"75e3cf52ad39647cd223bb86476ac180"},{url:"assets/js/page-记录生活.47671d47.js",revision:"ecfbf931fce08b8e7e8c8d92a466aefd"},{url:"assets/js/page-记录自己一些项目的开发文档.5a5c0559.js",revision:"45d0d7ca56ce019a961ab73a66b2b021"},{url:"assets/js/page-目标检测常用网络笔记.baac9bd8.js",revision:"cc98f6cad91c540abf0593e3be9a1080"},{url:"assets/js/page-算法可视化网页——水体提取.16dd3fe0.js",revision:"04cf60cc66cce68f7ec9b42199d9e70c"},{url:"assets/js/page-算法刷题笔记.ea9a8416.js",revision:"ad0d81dd47bddc9e0ad485cb3c20a8e8"},{url:"assets/js/page-雾霾天气查询网页.c7c9af59.js",revision:"ca898a4a3bf467defe5d81d7ab0c81b9"},{url:"assets/js/page-信号的采样.8b47dd2d.js",revision:"f83303656b8b1d0b61cc80375003c314"},{url:"assets/js/page-d2l目标检测笔记.fa682814.js",revision:"8d45cac548a7d0f8ad0860a86f0d0eb8"},{url:"assets/js/page-DSP,yyds!.d350ace6.js",revision:"6addecc0fb9ec29af348abd0c7826c24"},{url:"assets/js/page-FFT.9c99addb.js",revision:"8c363c814162bb78bad71332b5e51b9f"},{url:"assets/js/page-Handmack-Play播放器.8f228d8e.js",revision:"4977db23e0637b764c453c46dfd5b948"},{url:"assets/js/page-hereisNYC.5fb5505d.js",revision:"9bd971c000a31814dc6714b06c2af59c"},{url:"assets/js/page-Iloveboston!.5f9c3ee0.js",revision:"67b4d722a422cef6639c7dfe8e59f4c9"},{url:"assets/js/page-LDA.01cd4ea3.js",revision:"fa642ab35cce3f0d53c99b23d12c9dd8"},{url:"assets/js/page-Leetcode-19删除链表的倒数第n个结点.c055540c.js",revision:"51eb58126e472cc43b37cbfd3448ea32"},{url:"assets/js/page-Leetcode-370区间加法——差分数组java.8ca2e39b.js",revision:"c230b75c53f8c3092833fb194ee955f5"},{url:"assets/js/page-Leetcode-5691通过最少操作次数使数组的和相等.7705f144.js",revision:"81b79637f7353914456ea94b7719dc81"},{url:"assets/js/page-Leetcode-83删除链表重复元素——双指针cpp.548259d5.js",revision:"70bf9f6ee400518660adc4c35e161d6b"},{url:"assets/js/page-Mytour.585e2fee.js",revision:"2dbc87838d4ff86a4c7ccff7784a20db"},{url:"assets/js/page-notes!.a77ddf3a.js",revision:"2bdc2c0e956da0cfe7d4298d438287fd"},{url:"assets/js/page-PatternRecognition.2755d698.js",revision:"afd970375425a57445426c06385a704a"},{url:"assets/js/page-Report!.f07ba516.js",revision:"3e2d5b0c9bc53da912d6eebf7e3963b2"},{url:"assets/js/page-WelcometoCV!.13a495a4.js",revision:"33de278968c6ab328fb17ebd8e5b3559"},{url:"assets/js/vendors~flowchart.90a7d6f5.js",revision:"bea34cd6c2b772b171626e57246760fa"},{url:"assets/js/vendors~layout-Blog~layout-Layout.8d7dc64f.js",revision:"d646c45a2aedc1cd3fe32fc95bbf2fa0"},{url:"assets/js/vendors~layout-Blog~layout-Layout~layout-NotFound.cfcaf0fe.js",revision:"6fa48f45598c847273d1f33265cdf16c"},{url:"assets/js/vendors~layout-Blog~layout-Layout~layout-NotFound~layout-Slide.817296dc.js",revision:"e91702f7becb16a19536fb51b35c6a71"},{url:"assets/js/vendors~photo-swipe.2dc262c9.js",revision:"d77c6050c9dd278f5042b78ad3d309b6"},{url:"assets/js/vendors~reveal.df4bf573.js",revision:"e36797bb2e9da97f96692c0821a06352"},{url:"W.svg",revision:"cd83b8fdd2ac68c53cda43a2b82a2b61"},{url:"assets/fonts/KaTeX_AMS-Regular.2dbe16b4.ttf",revision:"2dbe16b4f4662798159f8d62c8d2509d"},{url:"assets/fonts/KaTeX_AMS-Regular.38a68f7d.woff2",revision:"38a68f7d18d292349a6e802a66136eae"},{url:"assets/fonts/KaTeX_AMS-Regular.7d307e83.woff",revision:"7d307e8337b9559e4040c5fb76819789"},{url:"assets/fonts/KaTeX_Caligraphic-Bold.33d26881.ttf",revision:"33d26881e4dd89321525c43b993f136c"},{url:"assets/fonts/KaTeX_Caligraphic-Regular.5e7940b4.ttf",revision:"5e7940b4ed250e98a512f520e39c867d"},{url:"assets/fonts/KaTeX_Fraktur-Bold.4de87d40.woff",revision:"4de87d40f0389255d975c69d45a0a7e7"},{url:"assets/fonts/KaTeX_Fraktur-Bold.7a3757c0.woff2",revision:"7a3757c0bfc580d91012d092ec8f06cb"},{url:"assets/fonts/KaTeX_Fraktur-Bold.ed330126.ttf",revision:"ed330126290a846bf0bb78f61aa6a080"},{url:"assets/fonts/KaTeX_Fraktur-Regular.450cc4d9.woff2",revision:"450cc4d9319c4a438dd00514efac941b"},{url:"assets/fonts/KaTeX_Fraktur-Regular.82d05fe2.ttf",revision:"82d05fe2abb0da9d1077110efada0f6e"},{url:"assets/fonts/KaTeX_Fraktur-Regular.dc4e330b.woff",revision:"dc4e330b6334767a16619c60d9ebce8c"},{url:"assets/fonts/KaTeX_Main-Bold.2e1915b1.ttf",revision:"2e1915b1a2f33c8ca9d1534193e934d7"},{url:"assets/fonts/KaTeX_Main-Bold.62c69756.woff",revision:"62c69756b3f1ca7b52fea2bea1030cd2"},{url:"assets/fonts/KaTeX_Main-Bold.78b0124f.woff2",revision:"78b0124fc13059862cfbe4c95ff68583"},{url:"assets/fonts/KaTeX_Main-BoldItalic.0d817b48.ttf",revision:"0d817b487b7fc993bda7dddba745d497"},{url:"assets/fonts/KaTeX_Main-BoldItalic.a2e3dcd2.woff",revision:"a2e3dcd2316f5002ee2b5f35614849a8"},{url:"assets/fonts/KaTeX_Main-BoldItalic.c7213ceb.woff2",revision:"c7213ceb79250c2ca46cc34ff3b1aa49"},{url:"assets/fonts/KaTeX_Main-Italic.081073fd.woff",revision:"081073fd6a7c66073ad231db887de944"},{url:"assets/fonts/KaTeX_Main-Italic.767e06e1.ttf",revision:"767e06e1df6abd16e092684bffa71c38"},{url:"assets/fonts/KaTeX_Main-Italic.eea32672.woff2",revision:"eea32672f64250e9d1dfb68177c20a26"},{url:"assets/fonts/KaTeX_Main-Regular.689bbe6b.ttf",revision:"689bbe6b67f22ffb51b15cc6cfa8facf"},{url:"assets/fonts/KaTeX_Main-Regular.756fad0d.woff",revision:"756fad0d6f3dff1062cfa951751d744c"},{url:"assets/fonts/KaTeX_Main-Regular.f30e3b21.woff2",revision:"f30e3b213e9a74cf7563b0c3ee878436"},{url:"assets/fonts/KaTeX_Math-BoldItalic.753ca3df.woff2",revision:"753ca3dfa44302604bec8e08412ad1c1"},{url:"assets/fonts/KaTeX_Math-BoldItalic.b3e80ff3.woff",revision:"b3e80ff3816595ffb07082257d30b24f"},{url:"assets/fonts/KaTeX_Math-BoldItalic.d9377b53.ttf",revision:"d9377b53f267ef7669fbcce45a74d4c7"},{url:"assets/fonts/KaTeX_Math-Italic.0343f93e.ttf",revision:"0343f93ed09558b81aaca43fc4386462"},{url:"assets/fonts/KaTeX_Math-Italic.2a39f382.woff2",revision:"2a39f3827133ad0aeb2087d10411cbf2"},{url:"assets/fonts/KaTeX_Math-Italic.67710bb2.woff",revision:"67710bb2357b8ee5c04d169dc440c69d"},{url:"assets/fonts/KaTeX_SansSerif-Bold.59b37733.woff2",revision:"59b3773389adfb2b21238892c08322ca"},{url:"assets/fonts/KaTeX_SansSerif-Bold.dfcc59ad.ttf",revision:"dfcc59ad994a0513b07ef3309b8b5159"},{url:"assets/fonts/KaTeX_SansSerif-Bold.f28c4fa2.woff",revision:"f28c4fa28f596796702fea3716d82052"},{url:"assets/fonts/KaTeX_SansSerif-Italic.3ab5188c.ttf",revision:"3ab5188c9aadedf425ea63c6b6568df7"},{url:"assets/fonts/KaTeX_SansSerif-Italic.99ad93a4.woff2",revision:"99ad93a4600c7b00b961d70943259032"},{url:"assets/fonts/KaTeX_SansSerif-Italic.9d0fdf5d.woff",revision:"9d0fdf5d7d27b0e3bdc740d90b40ec57"},{url:"assets/fonts/KaTeX_SansSerif-Regular.6c3bd5b5.woff",revision:"6c3bd5b57f7eba215a2d37e2e28077f1"},{url:"assets/fonts/KaTeX_SansSerif-Regular.badf3598.woff2",revision:"badf3598c22478fd9155a49391ecd396"},{url:"assets/fonts/KaTeX_SansSerif-Regular.d511ebce.ttf",revision:"d511ebcef253ab53775576f28315f350"},{url:"assets/fonts/KaTeX_Script-Regular.082640ca.ttf",revision:"082640ca4242bb2aade86855e4d7d5f6"},{url:"assets/fonts/KaTeX_Script-Regular.4edf4e0f.woff",revision:"4edf4e0fd49c8a5680dd541c05f16a4c"},{url:"assets/fonts/KaTeX_Script-Regular.af7bc98b.woff2",revision:"af7bc98b2200573686405dc784f53cf2"},{url:"assets/fonts/KaTeX_Size1-Regular.2c2dc3b0.ttf",revision:"2c2dc3b057bb48b80bc785ac3d87ecf8"},{url:"assets/fonts/KaTeX_Size2-Regular.114ad198.ttf",revision:"114ad19833311359052ad1a174159262"},{url:"assets/fonts/KaTeX_Size4-Regular.70174da7.ttf",revision:"70174da79d1707501c10e07872e84667"},{url:"assets/fonts/KaTeX_Typewriter-Regular.35fe2cce.ttf",revision:"35fe2cce0791c276b8e919decd873f5b"},{url:"assets/fonts/KaTeX_Typewriter-Regular.53dcf861.woff",revision:"53dcf861876aae6f3a6a6004dc3bc758"},{url:"assets/fonts/KaTeX_Typewriter-Regular.641339e2.woff2",revision:"641339e2cd86c7816bfb8028ee7f86e0"},{url:"404.html",revision:"d0cb5031abb6f66c61a7f8baec570a5b"},{url:"article/index.html",revision:"1f1be0ae6ab6215d00e4a2e1ba02538a"},{url:"category/后端/index.html",revision:"b2a7a8f5123e64c86353dd455e290cd5"},{url:"category/全栈开发/index.html",revision:"18d00669bad9cfa47df639a94abe03bb"},{url:"category/algorithm/index.html",revision:"ecdd4a2631a38e85aa99bc049e19268d"},{url:"category/cv/index.html",revision:"3e88fab90eb6d71756b59efac764931e"},{url:"category/dsp/index.html",revision:"1043eb556a1826f36d8fc4a9d054101e"},{url:"category/index.html",revision:"d74f4e8697fb17d803ed4cd4beb42a09"},{url:"develop/hm-play/Handmack-Play播放器/index.html",revision:"73bea28fb5c8733726e777b7cec56270"},{url:"develop/index.html",revision:"999ae83dd35d6b5ba8881e4d6a65c25d"},{url:"develop/water-app/水体提取/index.html",revision:"2acd5452fd2f15ed1cb6d1b539a8b63f"},{url:"develop/xdu-b_test/天气查询/index.html",revision:"adc693b2e512fc038e0355e368904835"},{url:"encrypt/index.html",revision:"d14fe9acce1e0e00c3e5fc64e7df1b0c"},{url:"index.html",revision:"74349ef8f67966a94286ea5bd67e27a9"},{url:"life/index.html",revision:"95832a59c6d925a74a2e0f2b8cadf39c"},{url:"life/tour/bos/index.html",revision:"62bae3c9a76b897dc8495c9841cb2bc3"},{url:"life/tour/index.html",revision:"20f51e588b155627bdbc58a4790adf29"},{url:"life/tour/ny/index.html",revision:"a2b0ededa26adee5871b0e47c33eec22"},{url:"notes/algorithm/index.html",revision:"fbfc6bb0ba94734726d6e1da668212ef"},{url:"notes/algorithm/Leetcode-19/index.html",revision:"74e5e06ef48ece425049994f2b2403d3"},{url:"notes/algorithm/Leetcode-370/index.html",revision:"28014d09b6e5b70e671a61cf0e37b3af"},{url:"notes/algorithm/Leetcode-5691/index.html",revision:"7056904ea1b69f907dc8dac03aebe58e"},{url:"notes/algorithm/Leetcode-83/index.html",revision:"70cdfaf279e1322aac171a614025e29d"},{url:"notes/cv/目标检测常用网络笔记/index.html",revision:"dae8ede96a5a19a4a14f5e07700c4ce8"},{url:"notes/cv/d2l目标检测笔记/index.html",revision:"ddc93eff624592bae350fed231e4808c"},{url:"notes/cv/index.html",revision:"a735048b6583839bf3e70dac41d5047b"},{url:"notes/index.html",revision:"a56f52fcdc52cd7cb8fc852987a2fffb"},{url:"reports/dsp/信号的产生与采样/index.html",revision:"244ccd8701a257a21ba9f9719a9d96f3"},{url:"reports/dsp/fft/index.html",revision:"537e15a0b6824ea93c6f350aff9c0c89"},{url:"reports/dsp/index.html",revision:"39e5fb219c1787bf7e96777d6d215cf0"},{url:"reports/index.html",revision:"e89a978031a357aa4019e2a543aba0e8"},{url:"reports/pr/ex1/index.html",revision:"291f046a76f5bf5f761e74ccf5ed4a19"},{url:"reports/pr/index.html",revision:"06fca5fa37c97d4e6a9258eff4cd4622"},{url:"slide/index.html",revision:"ed0b2fc2bad40d7883a8980f006d9d61"},{url:"tag/开发文档/index.html",revision:"db22fe20705d83ecee0e3257c3d3831b"},{url:"tag/目标检测/index.html",revision:"456b29d251dc15afb30e308901f0f875"},{url:"tag/手机端/index.html",revision:"c055ef595ece71d270a2363b2506c42e"},{url:"tag/数组/index.html",revision:"fd5e07cab67f23aa2015b35c979265ca"},{url:"tag/双指针/index.html",revision:"7b32baed656baf3caea359ef68c69c90"},{url:"tag/算法部署/index.html",revision:"d292b7ed07ac2821239cd472c53b5809"},{url:"tag/dsp/index.html",revision:"7a65afff337beca7817e5aef297e23f4"},{url:"tag/fft/index.html",revision:"3fe3a332ff479de430778ad2b900ce14"},{url:"tag/index.html",revision:"47ce1ca1648b29bc08718d5e0cf0ab8f"},{url:"tag/Leetcode/index.html",revision:"275987d4524720c7ce3f6db660d52775"},{url:"timeline/index.html",revision:"aeec47053edd46bb9b8c4b92214c7fe4"}],{}),e.cleanupOutdatedCaches()}));
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
