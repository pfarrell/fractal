import { ComplexNumber } from "./models.mjs";
self.onmessage = function (e) { 
	if (e.data !== undefined) { 
        for(let i=0; i<e.data.length; i++) { 
            var iter = new ComplexNumber(e.data[i].value.real, e.data[i].value.imaginary);
            var orig = iter;
            var iteration = 0;
            while(iteration < e.data[i].maxIterations && iter.real**2 + iter.imaginary**2 <= 4) {
                iteration+=1;
                iter = iter.iterate(orig);
            }
            e.data[i].iteration = iteration;
        }
        self.postMessage(e.data);
	} 
}
