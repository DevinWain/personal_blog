const { config } = require("vuepress-theme-hope");

module.exports = config({
    title: 'Wain\'s blog',
    description: 'Just playing around',
    themeConfig: {
    nav:[ // 导航栏配置
      {
        text: '首页', link: '/' 
      },
      {
        text: '生活记录', link: '/life/',
      },
      {
        text: '学习笔记', link: '/notes/',
      },
      {
        text: '课程报告', link: '/reports/',
      },
    ],
    mdEnhance: {
      enableAll: true,
    },
    sidebar: 'auto', // 侧边栏配置
    },
  });