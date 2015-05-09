 var generateHeatmap = function(data, id) {

    document.getElementById(id).innerHTML = ""
    // minimal heatmap instance configuration
    var heatmapInstance = h337.create({
      // only container is required, the rest will be defaults
      container: document.getElementById(id),
    });

    // now generate some random data
    var points = [];
    var max = 30;
    var width = 840;
    var height = 400;

    for (key in data) {
        var val = data[key][2];
        var point = {
          x: data[key][0] / 20 * height + 30,
          y: width/2 - (data[key][1] / 30 * width + 30),
          value: val
        };
        points.push(point);
    }

    // heatmap data format
    var data = {
      max: max,
      data: points
    };
    // if you have a set of datapoints always use setData instead of addData
    // for data initialization
    heatmapInstance.setData(data)
}
