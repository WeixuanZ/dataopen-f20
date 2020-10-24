mapboxgl.accessToken = 'pk.eyJ1Ijoid2VpeHVhbnoiLCJhIjoiY2tna3lkMnhjMWNubzMwdGVrN3ZtaHpqOSJ9.2ntrcdRtEDlAuW1OkflN4g';
var map = new mapboxgl.Map({
  container: 'map',
  style: 'mapbox://styles/mapbox/streets-v11',
  center: [-73.937724, 40.711725], // starting position [lng, lat]
  zoom: 9 // starting zoom
});