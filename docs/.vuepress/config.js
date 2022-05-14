const { config } = require("vuepress-theme-hope");
// https://mrhope.site/
module.exports = config({
    title: 'Wain\'s blog', 
    description: 'If you shed tears when you miss the sun, you also miss the stars.',
    head: [ // 注入到当前页面的 HTML <head> 中的标签
      ['link', { rel: 'icon', href: '/W.svg' }], // 增加一个自定义的网页标签的图标)
    ],
    themeConfig: {
      logo: "/W.svg",
      blog: {
        avatar: "/ava2.png",
        name: "Wain",
      },
    // https://vuepress-theme-hope.github.io/zh/guide/layout/navbar/#%E5%AF%BC%E8%88%AA%E6%A0%8F%E9%93%BE%E6%8E%A5
      nav:[ // 导航栏配置
        {
          text: '首页', link: '/' 
        },
        {
          text: '生活记录', prefix: "/life/", icon: "life",
          items: [
            {
              text: "生活记录", link: "", icon: "life",
            },
            { 
              text: "",
              items: [
                {text: "游记", link: "tour/"},
            
              ]
          },
          ],
        },
        {
          text: '学习笔记', prefix: '/notes/', icon: "note",
          items: [
            {
              text: '学习笔记', link: '', icon: "note",
            },
            { 
              text: "",
              items: [
                {text: "计算机视觉", link: "cv/"},
                {text: "算法笔记", link: "algorithm/"},
              ]
          },
          ]
        },
        {
          text: '开发文档', prefix: '/develop/', icon: "note",
          items: [
            {
              text: '开发文档', link: '', icon: "note",
            },
            { 
              text: "",
              items: [
                {text: "Handmack-Play播放器", link: "hm-play/Handmack-Play播放器"},
                {text: "B测-雾霾天气查询", link: "xdu-b_test/天气查询"},
                
              ]
          },
          ]
        },
        {
          text: '课程报告', prefix: '/reports/', icon: "paper",
          items: [
            {
              text: '课程报告', link: '', icon: "paper",
            },
            { 
              text: "",
              items:[
                {text: "数字信号处理", link: "dsp/",},
                {text: "模式识别", link: "pr/",}
              ] 
            },
          ]
        },
      ],
      mdEnhance: {
        enableAll: true,
        tasklist: true,
      },
    // https://vuepress-theme-hope.github.io/zh/guide/layout/sidebar/#%E6%A1%88%E4%BE%8B
      sidebar: {
        "/life/":[
          {
            title: "游记",
            prefix: "tour/",
            collapsable: false,
            children: ["bos", "ny"],
          }
        ],
        "/notes/":[
          {
            title: "计算机视觉",
            prefix: "cv/",
            collapsable: false,
            children: ["d2l目标检测笔记", "目标检测常用网络笔记"],
          },
          {
            title: "算法笔记",
            prefix: "algorithm/",
            collapsable: false,
            children: ["Leetcode-5691", "Leetcode-19", "Leetcode-83","Leetcode-370"],
          },
        ],
        "/develop/":[
          {
            title: "长期大project",
            prefix: "hm-play/",
            collapsable: false,
            children: ["Handmack-Play播放器"],
          },
          {
            title: "短期小项目",
            prefix: "xdu-b_test/",
            collapsable: false,
            children: ["天气查询"],
          }
        ],
        "/reports/":[
          {
            title: "数字信号处理",
            prefix: "dsp/",
            collapsable: false,
            children: ["信号的产生与采样", "fft"],
          },
          {
            title: "模式识别",
            prefix: "pr/",
            collapsable: false,
            children: ["ex1"],
          }
        ],
      }
    },
  });