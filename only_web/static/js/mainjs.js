 $(function () {
        $.ajax({url: "/load", async: true});
    });
upload_button = document.getElementById("upload_button")
text5 = document.getElementById("text5");
text6 = document.getElementById("text6");
$loading = $('#progress')
$progress = $('#progress-bar')
textarea1 = $('#textarea1')
textarea2 = $('#textarea2')
prg = 0
timer = 0

function openFileDialog()
{
$(".filebutton").click();
}

function fileSelected()
{
var fbutton = $(".filebutton")[0];//dom元素
//读取文件
var file = fbutton.files[0];
startFileUpload(file);
}

function openTextareaDialog()
{
    var subject = textarea1.html();
    if (subject.trim()!=='') {
        if(subject.trim().substring(0,2) === '1 '){
            textarea1.html('')
        }
        else {
            var fd = new FormData();
            fd.append("textarea", subject);
            var xhr = new XMLHttpRequest();
            xhr.open("POST", "/post/", false); //修改成自己的接口
            xhr.onreadystatechange = viewFunction;
            xhr.send(fd);
        }
    }
}

function startFileUpload(file)
{
var uploadURL = "/upload/";
//手工构造一个form对象
var formData = new FormData();
formData.append("fileToUpload" , file);// 'file' 为HTTP Post里的字段名, file 对浏览器里的File对象
//手工构造一个请求对象，用这个对象发送表单数据
//设置 progress, load, error, abort 4个事件处理器
var request = new XMLHttpRequest();
request.upload.addEventListener("progress" , uploadProgress , false);
request.addEventListener("load", uploadComplete, false);
request.addEventListener("error", uploadFailed, false);
request.addEventListener("abort", uploadCanceled, false);
request.onreadystatechange = viewFunction;
request.open("POST", uploadURL,true); // 设置服务URL
request.send(formData); // 发送表单数据
}

function viewFunction() {
     if(this.readyState === 4 && this.status === 200){
           textarea1.html(this.responseText);
        }
}

function uploadComplete(evt) {
    /* 服务器端返回响应时候触发event事件*/
    upload_button.innerHTML  = '上传成功';
}

function uploadProgress(evt) {
    if (evt.lengthComputable) {
        var percentComplete = Math.round(evt.loaded * 100 / evt.total);
        upload_button.innerHTML  = percentComplete.toString() + '%';
    } else {
        upload_button.innerHTML  = '上传失败';
    }
}

function uploadFailed(evt) {
    alert("上传失败");
}

function uploadCanceled(evt) {
    alert("上传取消");
}

function crawlers() {
    window.wxc.xcConfirm("目的地酒店名称","input",0)
                    }

function progress (dist, speed, delay, callback) {
		    var _dist = random(dist)
		    var _delay = random(delay)
		    var _speed = random(speed)
		    window.clearTimeout(timer)
		    timer = window.setTimeout(() => {
			if (prg + _speed >= _dist) {
			  window.clearTimeout(timer)
			  prg = _dist
			  callback && callback()
			} else {
			  prg += _speed
			  $progress.html(parseInt(prg) + '%')  // 留意，由于已经不是自增1，所以这里要取整
			  $progress.css("width" , parseInt(prg)+'%')  // 留意，由于已经不是自增1，所以这里要取整
			  console.log(prg)
			  progress(_dist, speed, delay, callback)
			}

		  }, _delay)
		}

function random (n) {
		  if (typeof n === 'object') {
			var times = n[1] - n[0]
			var offset = n[0]
			return Math.random() * times + offset
		  } else {
			return n
		  }
		}

function mycallback(data,now) {
            var endT = new Date();
            var mTime = (endT - now) / 1000;
            var Time = mTime.toFixed(2)
            var datalist = JSON.parse(data);
            $('#get_time').text(Time+'s')
            sessionStorage.setItem('datalist',JSON.stringify(datalist))
            echarts0 = echarts.init(document.getElementById('echarts-dataset0'), 'walden');
            option0 = {
                    // legend: {
                    //     data: ['positive', 'negative'],
                    //     left: 10,
                    // },
                    tooltip: {
                        trigger: 'axis',
                        axisPointer: {            // 坐标轴指示器，坐标轴触发有效
                            type: 'shadow'        // 默认为直线，可选为：'line' | 'shadow'
                        },
                        formatter: '{b0}<br/>{a0}: {c0}%<br />{a1}: {c1}%<br />'
                    },
                    grid: {
                        left: '3%',
                        right: '4%',
                        bottom: '80%',
                        containLabel: true
                    },
                    xAxis: [
                        {
                            splitLine: {show: false},
                            type: 'value',
                            inverse: true,
                            axisTick: {
                                show: false
                            },
                            axisLabel: {
                                formatter: '{value}%',
                                textStyle:{
                                    color:"rgba(255,255,255,0)",
                                    fontSize: 20
                                }
                            }
                        }
                    ],
                    yAxis: [
                        {
                            type: 'category',
                            axisTick: {
                                show: false
                            },
                            data: ['总体分析'],
                            axisLabel: {
                                show: false, formatter: '{value}%',
                                textStyle:{
                                    color:"rgb(255,255,255)",
                                    fontSize: 20
                                }
                            }
                        }
                    ],
                    series: [
                        {
                            name: 'positive',
                            type: 'bar',
                            stack: '总量',
                            label: {
                                show: true,
                                formatter: function (params) { //标签内容
                                    return params.data + '%'//,position: 'right'
                                }
                            },
                            data: [Math.round(datalist.chart1['positive'] * 10000) / 100]
                        },
                        {
                            name: 'negative',
                            type: 'bar',
                            stack: '总量',
                            itemStyle: {
                                normal: {
                                    color: '#1a6Dff'
                                }
                            },
                            label: {
                                show: true,
                                formatter: function (params) { //标签内容
                                    return params.data + '%'//,position: 'right'
                                }
                            },
                            data: [-Math.round(datalist.chart1['negative'] * 10000) / 100]
                        }
                    ]
                };
            echarts0.setOption(option0);
            echartsMap = echarts.init(document.getElementById('echarts-dataset1'), 'walden');
            function generate_data() {

                    var xAxisData = [];
                    var data1 = [];
                    var data2 = [];

                    $.each(datalist.chart2, function (k, v) {
                        data1.push(v.positive.toFixed(2));
                        data2.push((-v.negative).toFixed(2));
                        xAxisData.push(k);
                    })

                    var emphasisStyle = {
                        itemStyle: {
                            barBorderWidth: 1,
                            shadowBlur: 10,
                            shadowOffsetX: 0,
                            shadowOffsetY: 0,
                            shadowColor: 'rgba(0,0,0,0.5)'
                        }
                    };

                    optionDataset = {
                        title: {
                            text: '正负比例/一级指标',
                            x: 'center',
                            y: 'top',
                            textAlign: 'left',
                            textStyle:{
                                color: '#000000'
                            }
                        },
                        brush: {
                            toolbox: ['rect', 'polygon', 'lineX', 'lineY', 'keep', 'clear'],
                            xAxisIndex: 0
                        },
                        toolbox: {
                            feature: {
                                magicType: {
                                    type: ['stack', 'tiled']
                                },
                                dataView: {}
                            },
                            show: false
                        },
                        tooltip: {},
                        xAxis: {
                            data: xAxisData,
                            axisLine: {onZero: true},
                            splitLine: {show: false},
                            splitArea: {show: false},
                            axisLabel: {       //x轴标签
                                interval: 0,
                                //  rotate:40
                                textStyle:{
                                    color:"#000000",
                                    fontSize: 20
                                }
                            }
                        },
                        yAxis: {
                            inverse: false,
                            splitArea: {show: false},
                            max: 1,
                            min: -1,
                            axisTick: {       //y轴刻度线
                                "show": false
                            },
                            axisLabel: {       //y轴标签
                                "show": false,
                            }
                        },
                        grid: {
                            left: 0,
                        },
                        series: [
                            {
                                name: 'positive',
                                type: 'bar',
                                stack: 'one',
                                emphasis: emphasisStyle,
                                data: data1,
                                barWidth: 33
                            },
                            {
                                name: 'negative',
                                type: 'bar',
                                stack: 'one',
                                emphasis: emphasisStyle,
                                data: data2,
                                barWidth: 33,
                                itemStyle: {
                                    normal: {
                                        color: '#1a6Dff'
                                    }
                                }
                            }
                        ],
                        dataZoom: [
                                    {
                                        type: 'inside',//滑动轴
                                        start: 30,//距离左侧0刻度的距离，1%
                                        end: 70//距离左侧0刻度的距离，35%，相当于规定了滑动的范围
                                    }
                                ],
                    };
                    if (xAxisData.length >= 4) {
                        optionDataset.xAxis.axisLabel.interval = 0;
                        optionDataset.xAxis.axisLabel.rotate = 40;
                    }
                    return optionDataset;
                }
            optionMap = generate_data()
            echartsMap.setOption(optionMap);
            // echarts 窗口缩放自适应
            window.onresize = function () {
                    echarts0.resize()
                    echartsMap.resize();
                }
            echartsMap.getZr().off('click')
            echartsMap.getZr().on('click', params => {
                    let pointInPixel = [params.offsetX, params.offsetY]
                    if (echartsMap.containPixel('grid', pointInPixel)) {
                        let xIndex = echartsMap.convertFromPixel({seriesIndex: 0}, [params.offsetX, params.offsetY])[0]

                        var echartsMap2 = echarts.init(document.getElementById('echarts-dataset2'), 'walden');

                        function generate_data() {

                            var xAxisData = [];
                            var data1 = [];
                            var data2 = [];
                            var data3 = [];
                            chart_title = optionMap.xAxis.data[xIndex];

                            $.each(datalist.chart3[optionMap.xAxis.data[xIndex]], function (k, v) {
                                data1.push(v.positive.toFixed(2));
                                data2.push((-v.negative).toFixed(2));
                                data3.push(v.rate)
                                xAxisData.push(k);
                            })
                            data1.sort(function (a,b) {
                                 return  data3[data1.indexOf(b)] - data3[data1.indexOf(a)]
                            })
                            data2.sort(function (a,b) {
                                 return  data3[data2.indexOf(b)] - data3[data2.indexOf(a)]
                            })
                            xAxisData.sort(function (a,b) {
                                 return  data3[xAxisData.indexOf(b)] - data3[xAxisData.indexOf(a)]
                            })
                            data3.sort(function(a,b){return b-a})
                            var max_value = Math.max.apply(null, data3);
                            var max_index = data3.indexOf(max_value);
                            var emphasisStyle = {
                                itemStyle: {
                                    barBorderWidth: 1,
                                    shadowBlur: 10,
                                    shadowOffsetX: 0,
                                    shadowOffsetY: 0,
                                    shadowColor: 'rgba(0,0,0,0.5)'
                                }
                            };

                            optionDataset = {

                                title: {
                                    text: '二级指标：'+ chart_title,//'二级指标'
                                    x: 'center',
                                    y: 'top',
                                    textAlign: 'left',
                                    textStyle:{
                                    color: '#000000'
                                }
                                },
                                brush: {
                                    toolbox: ['rect', 'polygon', 'lineX', 'lineY', 'keep', 'clear'],
                                    xAxisIndex: 0
                                },
                                toolbox: {
                                    feature: {
                                        magicType: {
                                            type: ['stack', 'tiled']
                                        },
                                        dataView: {}
                                    },
                                    show: false
                                },
                                tooltip: {},
                                xAxis: {
                                    data: xAxisData,
                                    axisLine: {onZero: true},
                                    splitLine: {show: false},
                                    splitArea: {show: false},
                                    axisLabel: {       //x轴标签
                                        interval: 0,
                                        //  rotate:40
                                        textStyle:{
                                            color:"#000000",
                                            fontSize: 20
                                        }
                                    }
                                },
                                yAxis: {
                                    inverse: false,
                                    splitArea: {show: false},
                                    max: 1,
                                    min: -1,
                                    axisTick: {       //y轴刻度线
                                        "show": false
                                    },
                                    axisLabel: {       //y轴标签
                                        "show": false,
                                        textStyle:{
                                        color:"rgb(0,0,0)",
                                        fontSize: 20
                                }
                                    }
                                },
                                grid: {
                                    left: 0
                                },
                                series: [
                                    {
                                        name: 'positive',
                                        type: 'bar',
                                        stack: 'one',
                                        emphasis: emphasisStyle,
                                        data: data1,
                                        barWidth: 33,
                                        itemStyle: {
                                            normal:{
                                                color: function(params){
                                                    // var colorList = ['rgb(230,153,153)','rgb(195,53,49)']
                                                    // if (params.dataIndex === max_index)
                                                    //     return colorList[1]
                                                    // else
                                                    //     return colorList[0]
                                                    return 'rgb(195,53,49)'

                                                }
                                            }
                                        }
                                    },
                                    {
                                        name: 'negative',
                                        type: 'bar',
                                        stack: 'one',
                                        emphasis: emphasisStyle,
                                        data: data2,
                                        barWidth: 33,
                                        itemStyle: {
                                            normal: {
                                                color: function(params){
                                                // var colorList = ['rgb(111,230,214)','#1a6dff']
                                                // if (params.dataIndex === max_index)
                                                //     return colorList[1]
                                                // else
                                                //     return colorList[0]
                                                    return  '#1a6dff'
                                            }
                                            }
                                        }
                                    },
                                ],
                                dataZoom: [
                                    {
                                        type: 'inside',//滑动轴
                                        start: 30,//距离左侧0刻度的距离，1%
                                        end: 70//距离左侧0刻度的距离，35%，相当于规定了滑动的范围
                                    }
                                ],
                            };
                            if (xAxisData.length >= 4) {
                                optionDataset.xAxis.axisLabel.interval = 0;
                                optionDataset.xAxis.axisLabel.rotate = 40;
                            }
                            return optionDataset;
                        }
                        optionMap2 = generate_data()
                        echartsMap2.setOption(optionMap2, true);
                        window.onresize = function () {
                            echartsMap2.resize();
                        }
                        echartsMap2.getZr().off('click');
                        echartsMap2.getZr().on('click', params => {
                            let pointInPixel = [params.offsetX, params.offsetY]
                            if (echartsMap2.containPixel('grid', pointInPixel)) {
                                let xIndex = echartsMap2.convertFromPixel({seriesIndex: 0}, [params.offsetX, params.offsetY])[0]
                                third_chart_title = optionMap2.xAxis.data[xIndex]
                                evaluate = datalist.chart3[chart_title][third_chart_title]
                                text5.innerHTML = "";
                                text6.innerHTML = "";
                                $('#title_pos').html(third_chart_title + '的正面评价')
                                $('#title_neg').html(third_chart_title + '的反面评价')
                                for (i = 0; i < evaluate['positive_instances'].length; i++) {
                                    var para = document.createElement('a');//创建新的p标签
                                    var para0 = document.createElement('div');//创建新的div标签
                                    var para1 = document.createElement('div');//创建新的div标签
                                    var para2 = document.createElement('div');//创建新的div标签
                                    var node0 = document.createTextNode(evaluate['positive_instances'][i][0]);//创建一个文本节点
                                    var node1 = document.createTextNode(evaluate['positive_instances'][i][2]);//创建一个文本节点
                                    var node2 = document.createTextNode(evaluate['positive_instances'][i][1].length);//创建一个文本节点
                                    para0.appendChild(node0);//向p追加此文本节点
                                    para1.appendChild(node1);//向p追加此文本节点
                                    para2.appendChild(node2);//向p追加此文本节点
                                    para0.setAttribute('style','display:inline-block;width: 40%')
                                    para1.setAttribute('style','display:inline-block;width: 35%')
                                    para2.setAttribute('style','display:inline-block;width: 25%')
                                    para.appendChild(para0)
                                    para.appendChild(para1)
                                    para.appendChild(para2)
                                    para.className = "list-group-item";//向p追加className
                                    para.setAttribute("style","background-color:rgba(255,255,255,0);font-size:20px;margin-bottom:2px;font-weight: bold;font-color: black")
                                    para.setAttribute("role","button")
                                    para.setAttribute("data-toggle","popover")
                                    para.setAttribute("data-trigger","hover")
                                    para.setAttribute("title","用户观点：")
                                    para.setAttribute("data-content",evaluate['positive_instances'][i][1].join('<br>'))
                                    para.setAttribute("tabindex",i.toString())
                                    text5.appendChild(para);//向已有元素添加新元素（默认将para插入到element的最后）
                                }
                                for (i = 0; i < evaluate['negative_instances'].length; i++) {
                                    var para = document.createElement('a');//创建新的p标签
                                    var para0 = document.createElement('div');//创建新的div标签
                                    var para1 = document.createElement('div');//创建新的div标签
                                    var para2 = document.createElement('div');//创建新的div标签
                                    var node0 = document.createTextNode(evaluate['negative_instances'][i][0]);//创建一个文本节点
                                    var node1 = document.createTextNode(evaluate['negative_instances'][i][2]);//创建一个文本节点
                                    var node2 = document.createTextNode(evaluate['negative_instances'][i][1].length);//创建一个文本节点
                                    para0.appendChild(node0);//向p追加此文本节点
                                    para1.appendChild(node1);//向p追加此文本节点
                                    para2.appendChild(node2);//向p追加此文本节点
                                    para0.setAttribute('style','display:inline-block;width: 40%')
                                    para1.setAttribute('style','display:inline-block;width: 35%')
                                    para2.setAttribute('style','display:inline-block;width: 25%')
                                    para.appendChild(para0)
                                    para.appendChild(para1)
                                    para.appendChild(para2)
                                    para.className = "list-group-item";//向p追加className
                                    para.setAttribute("style","background-color:rgba(255,255,255,0);font-size:20px;margin-bottom:2px;font-weight: bold;font-color: black")
                                    para.setAttribute("role","button")
                                    para.setAttribute("data-toggle","popover")
                                    para.setAttribute("data-trigger","hover")
                                    para.setAttribute("title","用户观点：")
                                    para.setAttribute("data-content",evaluate['negative_instances'][i][1].join('<br>'))
                                    para.setAttribute("tabindex",i.toString())
                                    para.setAttribute("data-placement","left")
                                    text6.appendChild(para);//向已有元素添加新元素（默认将para插入到element的最后）
                                }
                            }
                            $('[data-toggle="popover"]').popover({
                                html : true
                            })
                            $('#evaluation_pos_show').show()
                            $('#evaluation_neg_show').show()
                        }
                        )//end chartclick
                    }
                })
    }

function model_count() {
    var now = new Date();
    $loading.show();
    progress([80, 90], [5, 10], 1000)  // 使用数组来表示随机数的区间
    $.ajax({
        url: "/mine",
        async: true,
        success:function(data) {
            mycallback(data,now)
            show_result();
        }
    })
}

function show_result() {
    var datalist = JSON.parse(sessionStorage.getItem('datalist'))
    // if (datalist == null) return;
    var showtext = ""
    for (let i = 0; i < datalist['text1'].length; i++) {
        let text_o = datalist['text2'][i]['text']
        let text = datalist['text2'][i]['text']
        for (let j = 0; j < datalist['text2'][i]['ap_span'].length; j++){
            let a = datalist['text2'][i]['ap_span'][j][0]
            let b = datalist['text2'][i]['ap_span'][j][1]
            let ap_span_o = text_o.substring(a, b + 1)
            if (ap_span_o.trim().length === 0){
                continue
            }
            let ap_span ="<span style='color: #a3110d'>" + ap_span_o + "</span>"
            text = text.replace(ap_span_o,ap_span)
        }
        for (let j = 0; j < datalist['text2'][i]['op_span'].length; j++){
            let a = datalist['text2'][i]['op_span'][j][0]
            let b = datalist['text2'][i]['op_span'][j][1]
            let op_span_o = text_o.substring(a, b + 1)
            if (op_span_o.trim().length === 0){
                continue
            }
            let op_span = "<span style='color: #032fdc'>" + op_span_o + "</span>"
            text = text.replace(op_span_o,op_span)
        }
        showtext = showtext + (i+1) + '   ' +text +"\n";
        showtext = showtext + datalist['text3'][i] +"\n";
        let exp_text = '';
        let exp_text_f = ' 事实: ';
        let exp_text_s = ' 建议：';
        let exp_text_r = ' 原因：';
        let exp_text_c = ' 条件： ';

        for (let j = 0; j < datalist['text2'][i]['express'].length; j++){
            let a = datalist['text2'][i]['express'][j][0][0];
            let b = datalist['text2'][i]['express'][j][0][1];
            let exp = text_o.substring(a,b+1);
            let kind_e = datalist['text2'][i]['express'][j][1];
            if (kind_e === 'fac'){
                exp_text_f = exp_text_f + exp + '   '
            }else if (kind_e === 'sug'){
                exp_text_s = exp_text_s + exp + '   '
            }else if (kind_e === 'rea'){
                exp_text_r = exp_text_r + exp + '   '
            }else if (kind_e === 'con'){
                exp_text_c = exp_text_c + exp + '   '
            }
        }
        exp_text = exp_text_f + exp_text_s + exp_text_r + exp_text_c
        showtext = showtext + '   ' +exp_text +'\n';
    }
    $('#textarea2').html(showtext);
    progress(100, [5, 15], 1, () => {
                                            window.setTimeout(() => {  // 延迟了一秒再隐藏loading
                                                $loading.hide()
                                                prg = 0
										        $progress.html(parseInt(prg) + '%')  // 留意，由于已经不是自增1，所以这里要取整
										        $progress.css("width" , parseInt(prg)+'%')  // 留意，由于已经不是自增1，所以这里要取整
                                            }, 1000)
                                        })
}

// function elementRe() {
//      var datalist = JSON.parse(sessionStorage.getItem('datalist'))
//      // if (datalist == null) return;
//      var showtext = ""
//      for (i = 0; i < datalist['text2'].length; i++) {
//          showtext = showtext + datalist['text2'][i] +"\n";
//      }
//      $('#textarea2').val(showtext);
//  }
//
//  function re() {
//     var datalist = JSON.parse(sessionStorage.getItem('datalist'))
//     // if (datalist == null) return;
//     var showtext = ""
//     for (i = 0; i < datalist['text3'].length; i++) {
//         showtext = showtext + datalist['text3'][i] +"\n";
//     }
//     $('#textarea2').val(showtext);
// }
//
// function polaritycla() {
//     var datalist = JSON.parse(sessionStorage.getItem('datalist'))
//     // if (datalist == null) return;
//     var showtext = ""
//     for (i = 0; i < datalist['text4'].length; i++) {
//         showtext = showtext + datalist['text4'][i] +"\n";
//     }
//     $('#textarea2').val(showtext);
// }
//
// function opex() {
//     var datalist = JSON.parse(sessionStorage.getItem('datalist'))
//     // if (datalist == null) return;
//     var showtext = ""
//     var fac = ""
//     var sug = ""
//     var con = ""
//     var rea = ""
//     if (datalist['chart4']['fac']){
//         for (i = 0; i < datalist['chart4']['fac'].length; i++) {
//         fac = fac + datalist['chart4']['fac'][i] +"\n";
//         }
//     }
//     if (datalist['chart4']['sug']) {
//         for (i = 0; i < datalist['chart4']['sug'].length; i++) {
//             sug = sug + datalist['chart4']['sug'][i] + "\n";
//         }
//     }
//     if (datalist['chart4']['con']) {
//         for (i = 0; i < datalist['chart4']['con'].length; i++) {
//             con = con + datalist['chart4']['con'][i] + "\n";
//         }
//     }
//     if (datalist['chart4']['rea']) {
//         for (i = 0; i < datalist['chart4']['rea'].length; i++) {
//             rea = rea + datalist['chart4']['rea'][i] + "\n";
//         }
//     }
//     showtext = "事实："+'\n'+fac+'\n'+"建议："+'\n'+sug+'\n'+"条件："+'\n'+con+"\n"+"原因："+'\n'+rea
//     $('#textarea2').val(showtext);
// }

function search() {
    var all_textarea_text = textarea1.html();
    var selected_text = all_textarea_text.substr(0, window.getSelection().focusOffset);
    var current_line_num = selected_text.split('\n').length - 1;
    var arry = all_textarea_text.split("\n");
    var selected_line_content = arry[current_line_num] +'\n';
    textarea1.html(all_textarea_text.replace(selected_line_content, '<div id = "selected_content" style="border:1px solid #f80707" onmouseout="clear_text()">'+selected_line_content+'</div>'));
    let selected_content = document.getElementById('selected_content')
    let move_dis = selected_content.offsetTop - selected_content.offsetHeight
    textarea2.scrollTop( move_dis * 3 -  selected_content.offsetHeight)
    all_textarea_text = textarea2.html();
    arry = all_textarea_text.split("\n");
    let selected_line_content_1 = arry[current_line_num * 3]
    let selected_line_content_2 = arry[current_line_num * 3 + 1]
    let selected_line_content_3 = arry[current_line_num * 3 + 2]
    selected_line_content = selected_line_content_1 + "\n" + selected_line_content_2+"\n"+selected_line_content_3 +"\n"
    textarea2.html(all_textarea_text.replace(selected_line_content, '<div  style="border:1px solid #f80707">'+selected_line_content+'</div>'));
}

function clear_text() {
    let reg = /<\/?div.*?>/g;
    var all_textarea_text = textarea1.html();
    all_textarea_text = all_textarea_text.replace(reg,'');
    textarea1.html(all_textarea_text)
    all_textarea_text = textarea2.html();
    all_textarea_text = all_textarea_text.replace(reg,'');
    textarea2.html(all_textarea_text)
}