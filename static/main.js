(function () {

    'use strict';

    var heatApp = angular.module('HeatApp', ['ngMaterial'])

    heatApp.controller('HeatAppController',
    ['$scope', '$log', '$http',
    function($scope, $log, $http) {

        $scope.getData = function() {
            $log.log("test");

            var userInput = $scope.input_url;

            $http.post('/', {"url": userInput})
                .success(function (results) {
                    $log.log(results);
                    $scope.results = results;
                })
                .error(function (error) {
                    $log.log(error);
                });
        };
    }

    ]);

    heatApp.directive('heatmap', ['$http', '$log', function ($http, $log) {

        return {
            template: '<div id="heatmapContainer"></div>',
            restrict: 'E',
            link: function (scope, element) {
                $http.post('/data', {"t":0})
                    .success(function (data) {
                        generateHeatmap(data, "heatmapContainer");
                    })
                    .error(function (error) {
                        $log.log(error);
                    });

                scope.$watch('t', function (v) {
                    if(v) {
                        $http.post('/data', {"t":v})
                            .success(function (data) {
                                generateHeatmap(data, "heatmapContainer");
                            })
                            .error(function (error) {
                                $log.log(error);
                            });
                    }
                });

            }
        }

    }]);

    heatApp.directive('heatmapSim', ['$http', '$log', function ($http, $log) {

        return {
            template: '<div id="heatmapContainerSim"></div>',
            restrict: 'E',
            link: function (scope, element) {
                $http.post('/data', {"t":0})
                    .success(function (data) {
                        generateHeatmap(data, "heatmapContainerSim");
                        scope.state = data;
                        $log.log(scope.state)
                        scope.last_t3 = 0;
                    })
                    .error(function (error) {
                        $log.log(error);
                    });

                scope.ahus = {
                    ahu_1_outlet: 20,
                    ahu_2_outlet: 20,
                    ahu_3_outlet: 20,
                    ahu_4_outlet: 20,
                };

                var update = function() {
                    scope.t3 = 0;
                    scope.last_t3 = 0;
                }

                var simulate = function(time_step, ahus, state) {
                    $http.post('/simulate',
                              {"t":time_step, "ahus":ahus, "state":state})
                        .success(function (data) {
                            generateHeatmap(data, "heatmapContainerSim");
                            scope.state = data;
                        })
                        .error(function (error) {
                            $log.log(error);
                        });
                };

                scope.$watch('ahus.ahu_1_outlet', function (v) {
                    update();
                });

                scope.$watch('ahus.ahu_2_outlet', function (v) {
                    update();
                });

                scope.$watch('ahus.ahu_3_outlet', function (v) {
                    update();
                });

                scope.$watch('ahus.ahu_4_outlet', function (v) {
                    update();
                });

                scope.$watch('t3', function (v) {
                    if(v <= scope.last_t3) {
                        v = scope.last_t3;
                        scope.t3 = scope.last_t3;
                    }
                    else {
                        simulate(v - scope.last_t3, scope.ahus, scope.state);
                        scope.last_t3 = v;
                    }
                });

            }
        }

    }]);

}())
