// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-search-agent-论文阅读记录",
        
          title: "Search Agent 论文阅读记录",
        
        description: "Paper reading notes on search agents, including query planning, retrieval, reflection, and evaluation",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/search-agent-paper-reading/";
          
        },
      },{id: "post-agent-ai-学习笔记",
        
          title: "Agent AI 学习笔记",
        
        description: "Notes from UC Berkeley CS294 Agentic AI course",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/CS294-Agentic-AI/";
          
        },
      },{id: "post-数学观察",
        
          title: "数学观察",
        
        description: "Mathematical observations including a geometric derivation of Lagrange multipliers",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/math-ob/";
          
        },
      },{id: "post-games101-笔记",
        
          title: "GAMES101 笔记",
        
        description: "GAMES101 lecture notes covering 3D Gaussian Splatting, MVP transforms, projection matrices, and viewport transforms",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/GAMES101/";
          
        },
      },{id: "post-zed-配置",
        
          title: "Zed 配置",
        
        description: "Zed editor configuration including settings, keybindings, and SSH/WSL setup",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/zed/";
          
        },
      },{id: "post-科研好物推荐",
        
          title: "科研好物推荐",
        
        description: "Useful tools for research including Semantic Scholar and DeepWiki",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/research-tool/";
          
        },
      },{id: "post-扩散模型学习",
        
          title: "扩散模型学习",
        
        description: "Learning notes on diffusion models covering DDPM, latent diffusion, classifier guidance, DiT, and ECGTwin",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/diffusion-model/";
          
        },
      },{id: "post-multimodal-reading",
        
          title: "Multimodal Reading",
        
        description: "Paper reading notes on multimodal models including Cascade-CLIP, DenseVLM, LOD, and DFormer",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/Multimodal-paper-reading/";
          
        },
      },{id: "post-心梗数据集整理",
        
          title: "心梗数据集整理",
        
        description: "Comprehensive survey of myocardial infarction datasets across ECG, MRI, CT, Echo, and multimodal sources",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/HeartAttack/";
          
        },
      },{id: "post-多模态模型hack研究",
        
          title: "多模态模型Hack研究",
        
        description: "Research log on adversarial attacks against multimodal models including CLIP, ALBEF, BLIP, and RGB-D fusion models",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/Hack-Multimodal/";
          
        },
      },{id: "post-vggt",
        
          title: "VGGT",
        
        description: "Notes on VGGT multi-scale geometry inference model architecture and design choices",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/VGGT/";
          
        },
      },{id: "post-llm推理框架笔记",
        
          title: "LLM推理框架笔记",
        
        description: "Notes on building LLM inference framework including CUDA kernel design for sgemv, mha, argmax, and more",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/infer-framework/";
          
        },
      },{id: "post-namomo-camp",
        
          title: "Namomo Camp",
        
        description: "2024 Namomo summer camp notes covering data structures, game theory, graph theory, math, and DP",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/Namomo-Camp/";
          
        },
      },{id: "post-observation",
        
          title: "Observation",
        
        description: "Competitive programming observations and problem-solving notes",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/ob/";
          
        },
      },{id: "post-my-summer-plan",
        
          title: "My summer plan",
        
        description: "Summer plan including gym schedule and competitive programming goals",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/summer-plan/";
          
        },
      },{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/assets/pdf/example_pdf.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
