mapboxgl.accessToken =
  "pk.eyJ1Ijoid2VpeHVhbnoiLCJhIjoiY2tna3lkMnhjMWNubzMwdGVrN3ZtaHpqOSJ9.2ntrcdRtEDlAuW1OkflN4g";
var map = new mapboxgl.Map({
  container: "map",
  style: "mapbox://styles/weixuanz/ckgo3bk222iew19quuk25vjwf",
  center: [-73.937724, 40.711725],
  zoom: 10,
});

var layerList = document.getElementById("menu");
var inputs = layerList.getElementsByTagName("input");

function switchLayer(layer) {
  var layerId = layer.target.value;
  map.setStyle("mapbox://styles/weixuanz/" + layerId);
}

for (var i = 0; i < inputs.length; i++) {
  inputs[i].onclick = switchLayer;
}
